
#' l1_p
#' @description Estimates non crossing quantile regression with a neural network.
#' @param X train predictor data
#' @param y train response data
#' @param test_X test predictor data
#' @param valid_X validation predictor data
#' @param tau target quantiles
#' @param hidden_dim1 the number of nodes in the first hidden layer
#' @param hidden_dim2 the number of nodes in the second hidden layer
#' @param learning_rate learning rate in the optimization process
#' @param max_deep_iter the number of iterations
#' @param lambda_obj the value of tuning parameter in the l1 penalization method
#' @param penalty the value of tuning parameter for ridge penalty on weights
#' @return y_predicted, y_test_predicted, y_valid_predited : predicted quantile based on train, test, and validation data, respectively

l1_p = function(X, y, test_X, valid_X, tau, hidden_dim1, hidden_dim2, learning_rate, max_deep_iter, lambda_obj, penalty = 0)
{
  input_dim = ncol(X)
  n = nrow(X)
  r = length(tau)
  p = hidden_dim2 + 1
  tau_mat = matrix(rep(tau, each = n), ncol = 1)
  # if tau is c(0.1,0.2), n=3, then tau_mat is c(0.1,0.1,0.1,0.2,0.2,0.2)

  input_x = tf$placeholder(tf$float32, shape(NULL, input_dim))  #
  output_y = tf$placeholder(tf$float32, shape(NULL, 1))         # (n, 1)
  output_y_tiled = tf$tile(output_y, shape(r, 1))               # (n * r, 1) repeat r times along the first dimension (rows) and 1 time along the second dimension (columns).
  # tf$tile: if a is [1,2], then tf$tile(a, (3,2)) is 3*4 matrix
  tau_tf = tf$placeholder(tf$float32, shape(n * r, 1))

  ### layer 1
  hidden_theta_1 = tf$Variable(tf$random_normal(shape(input_dim, hidden_dim1)))
  hidden_bias_1 = tf$Variable(tf$random_normal(shape(hidden_dim1)))
  hidden_layer_1 = tf$nn$sigmoid(tf$matmul(input_x, hidden_theta_1) + hidden_bias_1)

  ### layer 2
  hidden_theta_2 = tf$Variable(tf$random_normal(shape(hidden_dim1, hidden_dim2)))
  hidden_bias_2 = tf$Variable(tf$random_normal(shape(hidden_dim2)))
  feature_vec = tf$nn$sigmoid(tf$matmul(hidden_layer_1, hidden_theta_2) + hidden_bias_2) ##

  ### output layer
  delta_coef_mat = tf$Variable(tf$random_normal(shape(hidden_dim2, r)))
  delta_0_mat = tf$Variable(tf$random_normal(shape(1, r)))
  # delta_coef_mat and delta_0_mat are the weights and biases for the output layer, with r being the number of quantiles

  delta_mat = tf$concat(list(delta_0_mat, delta_coef_mat), axis = 0L)
  # delta_mat: (hidden_dim2+1,r)
  beta_mat = tf$transpose(tf$cumsum(tf$transpose(delta_mat)))
  # beta_mat: (hidden_dim2+1,r). (cumsum: [1,2,3] cumsum to [1,3,6]. Same dim)

  delta_vec = delta_mat[2:p, 2:r]                            # Delta matrix without the first row and column, shape: (hidden_dim2, r-1)
  delta_0_vec = delta_mat[1, 2:r ,drop = FALSE]              # First row of delta matrix without the first column, shape: (1, r-1)
  delta_minus_vec = tf$maximum(0, -delta_vec)                # Non-positive part of delta matrix, shape: (hidden_dim2, r-1)
  delta_minus_vec_sum = tf$reduce_sum(delta_minus_vec, 0L)   # Sum of non-positive deltas, shape: (r-1)
  delta_0_vec_clipped = tf$clip_by_value(delta_0_vec,
                                                     clip_value_min = tf$reshape(delta_minus_vec_sum, shape(nrow(delta_0_vec), ncol(delta_0_vec))),
                                                     clip_value_max = matrix(Inf, nrow(delta_0_vec), ncol(delta_0_vec)))  # Clipped delta values, shape: (1, r-1)
  # tf$clip_by_value is a function in TensorFlow that clips (or limits) the values of a tensor to be within a specified range. 
  # It ensures that all elements in the tensor fall between the specified minimum and maximum values.

  #### optimization
  delta_constraint = delta_0_vec_clipped - delta_minus_vec_sum
  delta_clipped = tf$clip_by_value(delta_constraint, clip_value_min = 10e-20, clip_value_max = Inf)
  # Those two are not used...???

  predicted_y_modified = tf$matmul(feature_vec, beta_mat[2:p, ]) +
    tf$cumsum(tf$concat(list(beta_mat[1, 1, drop = FALSE], delta_0_vec_clipped), axis = 1L), axis = 1L) # not used in the training step
  predicted_y = tf$matmul(feature_vec, beta_mat[2:p, ]) + beta_mat[1, ] # use the updated parameters to get predicted_y
  predicted_y_tiled = tf$reshape(tf$transpose(predicted_y), shape(n * r, 1))

  diff_y = output_y_tiled - predicted_y_tiled
  quantile_loss = tf$reduce_mean(diff_y * (tau_tf - (tf$sign(-diff_y) + 1)/2 ))

  objective_fun = quantile_loss +
    penalty * (tf$reduce_mean(hidden_theta_1^2) + tf$reduce_mean(hidden_theta_2^2) +
                 tf$reduce_mean(delta_coef_mat^2)) +
    lambda_obj * tf$reduce_mean(tf$abs(delta_0_vec - delta_0_vec_clipped))

  train_opt = tf$train$RMSPropOptimizer(learning_rate = learning_rate)$minimize(objective_fun)  # RMSProp optimizer for minimizing the objective function.

  sess = tf$Session()                          # Create a TensorFlow session
  sess$run(tf$global_variables_initializer())  # Initialize all variables

  tmp_vec = numeric(max_deep_iter)
  for(step in 1:max_deep_iter)                 # Training loop for max_deep_iter iterations
  {
    sess$run(train_opt,
             feed_dict = dict(input_x = X,
                              output_y = y,
                              tau_tf = tau_mat))
  }

  y_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = X))
  y_test_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = test_X))
  y_valid_predict = sess$run(predicted_y_modified, feed_dict = dict(input_x = valid_X))

  sess$close()
  barrier_result = list(y_predict = y_predict, y_valid_predict = y_valid_predict, y_test_predict = y_test_predict)
  return(barrier_result)
}

