library(SharedForest)

## Load library
library(SoftBart)
library(bartMachine)


## Functions used to generate fake data
set.seed(1234)
f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
  10 * x[,4] + 5 * x[,5]

gen_data_fried <- function(n_train, P, sigma) {
  X <- matrix(runif(n_train * P), nrow = n_train)
  mu <- f_fried(X)
  
  Y <- mu + sigma * rnorm(n_train)
  Z <- mu + sigma * rnorm(n_train)
  delta <- ifelse(Z > 0, 1, 0)
  
  return(list(X = X, Y = Y, W = W, delta = delta, mu = mu))
}

gen_data <- function(n_train, n_test, P, sigma) {
  X <- matrix(runif(n_train * P), nrow = n_train)
  mu <- f_fried(X)
  X_test <- matrix(runif(n_test * P), nrow = n_test)
  mu_test <- f_fried(X_test)
  Y <- mu + sigma * rnorm(n_train)
  Y_test <- mu_test + sigma * rnorm(n_test)
  
  return(list(X = X, Y = Y, mu = mu, X_test = X_test, Y_test = Y_test, mu_test = mu_test))
}

gen_data_2 <- function(n) {
  X <- matrix(runif(n * 2), nrow = n)
  eta <- X[,1] + X[,2]
  mu <- eta
  sigma <- .2 + .3 * abs(eta)
  Y <- mu + sigma * rnorm(n)
  return(list(X = X, Y = Y, mu = mu, sigma = sigma))
}

gen_data_3 <- function(n) {
  X <- matrix(rnorm(n * 2), nrow = n)
  eta <- X[,1] + X[,2]
  mu <- eta
  sigma <- .2 + .3 * abs(eta)
  Y <- mu + sigma * rnorm(n)
  Z <- ifelse(rnorm(n,eta) < 0, 0, 1)
  return(list(X = X, Y = Y, delta = Z, W = X, mu = mu, sigma = sigma))
}

## Simiulate dataset
# sim_data <- gen_data(250, 100, 5, 1)
# sim_data <- gen_data_2(1000)
sim_data <- gen_data_3(200)

X <- sim_data$X
Y <- sim_data$Y
delta <- sim_data$delta
W <- sim_data$W

X <- quantile_normalize_bart(X)
W <- X
Y <- scale(Y)
mu_Y <- attributes(Y)[[2]]
sigma_Y <- attributes(Y)[[3]]

a <- 1.5
num_tree <- 50
f <- function(x) {
  alpha <- x[1]
  beta <- x[2]
  mu <- digamma(alpha) - log(beta)
  sigma <- sqrt(trigamma(alpha))
  return(mu^2 + (sigma - sqrt(a / num_tree))^2)
}
hypers <- Hypers(X,Y, num_tree = num_tree)
opts <- Opts(num_burn = 1000)

pars <- optim(c(1,1), f)$par
hypers$a_tau <- pars[1]
hypers$b_tau <- pars[2]
hypers$kappa <- 0.5 * hypers$num_tree
hypers$sigma_hat <- GetSigma(X,Y)
hypers$sigma_theta <- 3 / sqrt(2 * hypers$num_tree)
hypers$theta_0 <- qnorm(mean(delta))

foo <- SharedBart(X = X, Y = Y, W = W, delta = delta, hypers_ = hypers, opts_ = opts)

foo_bm <- bartMachine(as.data.frame(sim_data$X), sim_data$Y)
foo_cm <- bartMachine(as.data.frame(sim_data$X), as.factor(sim_data$delta))

plot(colMeans(foo$mu_hat[-(1:2500),]), (sim_data$mu - mu_Y) / sigma_Y)
mu <- sim_data$mu
mu_hat <- sigma_Y * colMeans(foo$mu_hat[-(1:500),]) + mu_Y
mu_bm <- predict(foo_bm, as.data.frame(sim_data$X))
theta_bm <- predict(foo_cm, as.data.frame(sim_data$X))
rmse <- function(x,y) sqrt(mean((x-y)^2))
rmse(mu_hat, mu)
rmse(mu_bm, mu)
plot(mu_bm, mu)
plot(mu_hat, mu)
cor(colMeans(foo$theta_hat), mu)
cor(mu_bm, mu)^2
cor(theta_bm, mu)^2
cor(mu_hat, mu)
plot(sigma_Y * colMeans(1.0 / sqrt(foo$tau_hat)))
