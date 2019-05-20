library(SharedForest)
library(tidyverse)
options(java.parameters = "-Xmx8g")
library(bartMachine)
library(dbarts)



## Functions used to generate fake data
# set.seed(123)
f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
  10 * x[,4] + 5 * x[,5]

f_fried2 <- function(x) ## Friedman's five dimensional test function
  0.5*sin(pi*x[ , 1]*x[ , 2])+(x[ , 3]-0.5)^2+0.5*x[ , 4]+0.25*x[ , 5]-1

gen_data_fried <- function(n_train, P, sigma) {
  X <- matrix(runif(n_train * P), nrow = n_train)
  mu <- f_fried(X)
  theta <- f_fried2(X)
  
  Y <- mu + sigma * rnorm(n_train)

  delta <- ifelse(rnorm(n_train, theta) < 0, 0, 1)

  return(list(X = X, Y = Y, W = X, delta = delta, mu = mu, theta = theta))
}

gen_data_3 <- function(n) {
  X <- matrix(rnorm(n * 2), nrow = n)
  eta <- X[,1] + X[,2]
  mu <- eta
  sigma <- .2 + .3 * abs(eta)
  Y <- mu + sigma * rnorm(n)
  Z <- ifelse(rnorm(n,eta) < 0, 0, 1)
  return(list(X = X, Y = Y, delta = Z, W = X, mu = mu, sigma = sigma, theta = eta))
}


set.seed(1234)
datum <- gen_data_fried(200, 180, 1)
# datum <- gen_data_3(1000)
# datum$X <- pnorm(datum$X)
# datum$W <- pnorm(datum$W)
Y_scale <- scale(datum$Y)
mu_Y <- attributes(Y_scale)[[2]]
scale_Y <- attributes(Y_scale)[[3]]

hypers <- Hypers(X = datum$X, Y = datum$Y, W = datum$W, delta = datum$delta, k_theta = 4)
opts <- Opts(num_print = 10)

  

fit_n <- with(datum, SharedBart(X = matrix(nrow=0,ncol=ncol(X)), Y = numeric(0),
                                W = W, delta = delta, hypers_ = hypers, opts_ = opts))
fit <- with(datum, SharedBart(X = X, Y = Y_scale, W = W, delta = delta, hypers_ = hypers, opts_ = opts))
fit_n2 <- with(datum, SharedBart(X = X, Y=Y_scale, W = matrix(nrow=0,ncol=ncol(X)), delta = numeric(0), hypers_ = hypers, opts_ = opts))
fit_cm <- bartMachine(X = as.data.frame(datum$W), y = as.factor(1 - datum$delta))
fit_bm <- bartMachine(X = as.data.frame(datum$X), y = datum$Y)
fit_db <- bart(datum$W, datum$delta)
fit_dbr <- bart(datum$X, datum$Y)

rmse <- function(x,y) sqrt(mean((x-y)^2))

cor_theta <- c(
  cor(as.numeric(fit_n$theta_hat_mean), datum$theta),
  cor(as.numeric(fit$theta_hat_mean), datum$theta),
  cor(colMeans(fit_db$yhat.train), datum$theta), 
  cor(qnorm(fit_cm$p_hat_train), datum$theta)
)

rmse_theta <- c(
  rmse(as.numeric(fit_n$theta_hat_mean), datum$theta),
  rmse(as.numeric(fit$theta_hat_mean), datum$theta),
  rmse(colMeans(fit_db$yhat.train), datum$theta),
  rmse((qnorm(fit_cm$p_hat_train)), datum$theta)
)

rmse_mu <- c(
  rmse(as.numeric(fit_n2$mu_hat_mean) * scale_Y + mu_Y, datum$mu),
  rmse(as.numeric(fit$mu_hat_mean) * scale_Y + mu_Y, datum$mu),
  rmse((fit_dbr$yhat.train.mean), datum$mu), 
  rmse(fit_bm$y_hat_train, datum$mu)
)

out <- rbind(cor_theta, rmse_theta, rmse_mu)

colnames(out) <- c("solo", "shared", "dbarts", "bm")
rownames(out) <- c("cor_theta", "rmse_theta", "rmse_mu")

out

ggdf <- tibble(
  theta = c(
    fit$theta_hat_mean, 
    fit_n$theta_hat_mean, 
    colMeans(fit_db$yhat.train), 
    qnorm(fit_cm$p_hat_train),
    datum$theta), 
  method = c(
    rep("shared", length(fit$theta_hat_mean)), 
    rep("noshared", length(fit_n$theta_hat_mean)), 
    rep("dbarts", ncol(fit_db$yhat.train)), 
    rep("bm", length(fit_cm$p_hat_train)),
    rep("truth", length(datum$theta))
  ))

ggplot(ggdf, aes(x = theta, color = method)) + geom_density() + theme_bw() + facet_wrap(~method)

plot(fit$theta_hat_mean, datum$theta, asp = 1)
abline(a=0,b=1)
plot(fit_n$theta_hat_mean, datum$theta, asp = 1)
abline(a=0,b=1)
