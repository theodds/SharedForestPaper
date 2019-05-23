#' Create hyperparameter object for SoftBart
#'
#' Creates a list which holds all the hyperparameters for use with the softbart
#' command.
#'
#' @param X NxP matrix of training data covariates.
#' @param Y Nx1 vector of training data response.
#' @param group For each column of X, gives the associated group
#' @param alpha Positive constant controlling the sparsity level
#' @param beta Parameter penalizing tree depth in the branching process prior
#' @param gamma Parameter penalizing new nodes in the branching process prior
#' @param k Related to the signal-to-noise ratio, sigma_mu = 0.5 / (sqrt(num_tree) * k). BART defaults to k = 2.
#' @param sigma_hat A prior guess at the conditional variance of Y. If not provided, this is estimated empirically by linear regression.
#' @param shape Shape parameter for gating probabilities
#' @param width Bandwidth of gating probabilities
#' @param num_tree Number of trees in the ensemble
#' @param alpha_scale Scale of the prior for alpha; if not provided, defaults to P
#' @param alpha_shape_1 Shape parameter for prior on alpha; if not provided, defaults to 0.5
#' @param alpha_shape_2 Shape parameter for prior on alpha; if not provided, defaults to 1.0
#' @param num_tree_prob Parameter for geometric prior on number of tree
#'
#' @return Returns a list containing the function arguments.
Hypers <- function(X,Y,W,delta, group = NULL,
                   alpha = 1,
                   beta = 2,
                   gamma = 0.95,
                   num_tree = 50,
                   var_tau = 0.5,
                   k = 2, ## Determines kappa
                   k_theta = 2,
                   alpha_scale = NULL,
                   alpha_shape_1 = 0.5,
                   alpha_shape_2 = 1,
                   sigma_hat = NULL, ## Determines tau_0 and its prior scale
                   sigma_theta = NULL,
                   theta_0 = NULL,
                   shape = 1) {


  ## Preprocess stuff (in order they appear in args)
  Y <- scale(Y)

  if(is.null(group)) {
    group                          <- 1:ncol(X) - 1
  } else {
    group                          <- group - 1
  }

  obj <- function(x) {
    alpha <- x[1]
    beta <- x[2]
    mu <- digamma(alpha) - log(beta)
    sigma <- sqrt(trigamma(alpha))
    return(mu^2 + (sigma - sqrt(var_tau / num_tree))^2)
  }
  pars <- optim(c(1,1), obj)$par
  a_tau <- pars[1]
  b_tau <- pars[2]

  alpha_scale <- ifelse(is.null(alpha_scale), ncol(X), alpha_scale)
  sigma_hat <- ifelse(is.null(sigma_hat), GetSigma(X,Y), sigma_hat)
  theta_0   <- ifelse(is.null(theta_0), qnorm(mean(delta)), theta_0)

  ## MAKE OUTPUT AND RETURN IT

  out                                  <- list()

  out$alpha                            <- alpha
  out$beta                             <- beta
  out$gamma                            <- gamma
  out$num_tree                         <- num_tree
  out$a_tau                            <- a_tau
  out$b_tau                            <- b_tau
  out$kappa                            <- 1.0 / (3.0 / (k * sqrt(num_tree)))^2
  out$alpha_scale                      <- alpha_scale
  out$alpha_shape_1                    <- alpha_shape_1
  out$alpha_shape_2                    <- alpha_shape_2
  out$sigma_hat                        <- sigma_hat
  out$sigma_theta                      <- 3.0 / (k_theta * sqrt(num_tree))
  out$theta_0                          <- theta_0
  out$group                            <- group



  return(out)

}

#' MCMC options for SoftBart
#'
#' Creates a list which provides the parameters for running the Markov chain.
#'
#' @param num_burn Number of warmup iterations for the chain.
#' @param num_thin Thinning interval for the chain.
#' @param num_save The number of samples to collect; in total, num_burn + num_save * num_thin iterations are run
#' @param num_print Interval for how often to print the chain's progress
#' @param update_sigma_mu If true, sigma_mu/k are updated, with a half-Cauchy prior on sigma_mu centered at the initial guess
#' @param update_s If true, s is updated using the Dirichlet prior.
#' @param update_alpha If true, alpha is updated using a scaled beta prime prior
#' @param update_beta If true, beta is updated using a Normal(0,2^2) prior
#' @param update_gamma If true, gamma is updated using a Uniform(0.5, 1) prior
#' @param update_tau If true, tau is updated for each tree
#' @param update_tau_mean If true, the mean of tau is updated
#'
#' @return Returns a list containing the function arguments
Opts <- function(num_burn = 2500,
                 num_thin = 1,
                 num_save = 2500,
                 num_print = 100,
                 update_sigma_mu = TRUE,
                 update_s = TRUE,
                 update_alpha = TRUE) {
  out <- list()
  out$num_burn        <- num_burn
  out$num_thin        <- num_thin
  out$num_save        <- num_save
  out$num_print       <- num_print
  out$update_s        <- update_s
  out$update_alpha    <- update_alpha
  # out$update_num_tree <- update_num_tree
  out$update_num_tree <- FALSE

  return(out)

}

GetSigma <- function(X,Y) {

  stopifnot(is.matrix(X) | is.data.frame(X))

  if(is.data.frame(X)) {
    X <- model.matrix(~.-1, data = X)
  }


  fit <- cv.glmnet(x = X, y = Y)
  fitted <- predict(fit, X)
  sigma_hat <- sqrt(mean((fitted - Y)^2))

  return(sigma_hat)
}
