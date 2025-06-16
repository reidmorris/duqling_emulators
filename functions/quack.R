quack <- function(function_name, n_samples, seed = NULL, ...) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  tryCatch({
    func_info <- duqling::quack(function_name)
  }, error = function(e) {
    stop(paste("Function", function_name, "not found in duqling package"))
  })

  input_dim <- func_info$input_dim

  X <- matrix(runif(n_samples * input_dim), nrow = n_samples, ncol = input_dim)
  y <- duqling::duq(X, function_name, scale01 = TRUE, ...)

  list(X = X, y = y, func_info = func_info)
}