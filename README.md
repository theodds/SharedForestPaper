README
================



Install
-------

There are two packages included in this repository: `GammaForest` and `SharedForest`. After downloading this repository, the first step is to install both of these packages by running (say)

    R CMD INSTALL ./GammaForest/
    R CMD INSTALL ./SharedForest/

I have tested that these packages can be installed on Mac OSX and Linux, and it is assumed that interested users will be able to figure out how to overcome any issues installing these packages (see www.github.com/theodds/SoftBART for a hint on installing on Mac, everything should be just-work on Ubuntu if you have the appropriate dependencies installed).

Status of developement
----------------------

The code provided here is being actively developed, and not all aspects of the code have been properly documented. It is advised that users stick to default hyperparameter specification unless they *really* know what they are doing.

Heteroskedastic normal/probit models
------------------------------------

The `SharedForest` package fits the following heteroskedastic normal/probit model with

![
\\begin{aligned}
  Y\_i      &\\sim {\\operatorname{Normal}}(f(X\_i), \\sigma^2(X\_i)), \\\\
  \\delta\_k &\\sim {\\operatorname{Bernoulli}}(h(W\_k)).
\\end{aligned}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Baligned%7D%0A%20%20Y_i%20%20%20%20%20%20%26%5Csim%20%7B%5Coperatorname%7BNormal%7D%7D%28f%28X_i%29%2C%20%5Csigma%5E2%28X_i%29%29%2C%20%5C%5C%0A%20%20%5Cdelta_k%20%26%5Csim%20%7B%5Coperatorname%7BBernoulli%7D%7D%28h%28W_k%29%29.%0A%5Cend%7Baligned%7D%0A "
\begin{aligned}
  Y_i      &\sim {\operatorname{Normal}}(f(X_i), \sigma^2(X_i)), \\
  \delta_k &\sim {\operatorname{Bernoulli}}(h(W_k)).
\end{aligned}
")

 where it is assumed that ![X\_i](https://latex.codecogs.com/png.latex?X_i "X_i") and ![W\_k](https://latex.codecogs.com/png.latex?W_k "W_k") consist of the same predictors (so that it makes sense to share the forest across these responses). The zero-inflated log-normal model can then be represented by (i) taking the logarithm of the response for ![Y\_i](https://latex.codecogs.com/png.latex?Y_i "Y_i") (if it is non-zero) and (ii) taking ![\\delta\_k](https://latex.codecogs.com/png.latex?%5Cdelta_k "\delta_k") to be the indicator that the response is non-zero.

First, load the package:

``` r
library(SharedForest)
```

The convenience function `Hypers` produces a list which gives the hyperparameters of the model. This can be obtained as

``` r
hypers <- Hypers(X = X, Y = Y_scale, W = W, delta = delta)
```

Notice that the response `Y` has been assumed to have been scaled to mean 0 and standard deviation 1. The options for running the chain can be set using the convenience functions `Opts` as

``` r
opts <- Opts(num_burn = 2000, num_thin = 1, num_save = 2000, num_print = 10)
```

Crucially, we also require the predictors to also be scaled to lie in ![\[0,1\]](https://latex.codecogs.com/png.latex?%5B0%2C1%5D "[0,1]"). A simple way to do this is using the `SoftBart` package (available [here](www.github.com/theodds/SoftBART)) to perform quantile normalization:

``` r
X_norm <- SoftBart::quantile_normalize_bart(X)
W_norm <- SoftBart::quantile_normalize_bart(W)
```

We can then fit the model as

``` r
fit_shared <- SharedBart(X = X_norm, Y = Y_scale, W = W_norm, delta = delta, 
                         X_test = X_norm, W_test = W_norm, hypers_ = hypers, 
                         opts_ = opts)
```

Shared Forest Gamma Models
--------------------------

The `GammaForest` package fits the gamma/probit model with

![
\\begin{aligned}
  Y\_i      &\\sim {\\operatorname{Gam}}(\\alpha, \\alpha f(X\_i)), \\\\ 
  \\delta\_k &\\sim {\\operatorname{Bernoulli}}(h(W\_k)).
\\end{aligned}
](https://latex.codecogs.com/png.latex?%0A%5Cbegin%7Baligned%7D%0A%20%20Y_i%20%20%20%20%20%20%26%5Csim%20%7B%5Coperatorname%7BGam%7D%7D%28%5Calpha%2C%20%5Calpha%20f%28X_i%29%29%2C%20%5C%5C%20%0A%20%20%5Cdelta_k%20%26%5Csim%20%7B%5Coperatorname%7BBernoulli%7D%7D%28h%28W_k%29%29.%0A%5Cend%7Baligned%7D%0A "
\begin{aligned}
  Y_i      &\sim {\operatorname{Gam}}(\alpha, \alpha f(X_i)), \\ 
  \delta_k &\sim {\operatorname{Bernoulli}}(h(W_k)).
\end{aligned}
")

As before, we assume that ![X\_i](https://latex.codecogs.com/png.latex?X_i "X_i") and ![W\_k](https://latex.codecogs.com/png.latex?W_k "W_k") consist of the same predictors. The zero-inflated gamma model can be represented by taking ![\\delta\_k](https://latex.codecogs.com/png.latex?%5Cdelta_k "\delta_k") to be the indicator that ![Y\_i](https://latex.codecogs.com/png.latex?Y_i "Y_i") is not equal to zero; here, there is no need to perform the log transformation.

The interfact here is basically the same as with shared forest. We can run

``` r
library(GammaForest)
fit_gamma <- GammaBart(X = X_norm, Y = Y_scale, W = W_norm, delta = delta, 
                       X_test = X_norm, W_test = W_norm, hypers_ = hypers, 
                       opts_ = opts)
```

The only subtlety is that the `Hypers` function in the `GammaForest` package has slightly different hyperparameters, and so `hypers` should be constructed with \``GammaForest::Hypers`.

An Analysis
-----------

We illustrate the use of `SharedForest` under the same simulation settings as in the paper. The following code generates data.

``` r
gen_data_fried <- function(n, P,sigma = 1, sigma_theta = 1) {
  f_fried <- function(x) 10 * sin(pi * x[,1] * x[,2]) + 20 * (x[,3] - 0.5)^2 + 
               10 * x[,4] + 5 * x[,5]
  X <- matrix(runif(n * P), nrow = n)
  mu <- f_fried(X)
  theta <- sigma_theta * (f_fried(X) / 20 - .7)
  
  Y <- mu + sigma * rnorm(n)
  
  delta <- ifelse(rnorm(n, theta) < 0, 0, 1)
  
  return(list(X = X, Y = Y, W = X, delta = delta, mu = mu, theta = theta))
}
```

The following code fits the shared model, as well as a probit model to ![\\delta](https://latex.codecogs.com/png.latex?%5Cdelta "\delta") which does not share.

``` r
library(zeallot)
library(SharedForest)

set.seed(7554450)
c(X,Y,W,delta,mu,theta)            %<-% gen_data_fried(250, 100)
c(Xt, Yt, Wt, deltat, mut, thetat) %<-% gen_data_fried(250,100)

hypers <- SharedForest::Hypers(X = X, Y = Y, W = W, delta = delta)
opts   <- SharedForest::Opts()

fit_shared  <- SharedBart(X = X, Y = Y, W = W, delta = delta, X_test = Xt, 
                          W_test = Wt, hypers_ = hypers, opts_ = opts)
fit_noshare <- SharedBart(X = X[0,], Y = Y[0], W = W, delta = delta, 
                          X_test = Xt[0,], W_test = Wt, hypers_ = hypers, 
                          opts_ = opts)
```

We then compare the quality of the fit using the average KL-divergence from the estimated probability of ![\\delta\_i = 1](https://latex.codecogs.com/png.latex?%5Cdelta_i%20%3D%201 "\delta_i = 1") to the truth on the held-out data (lower is better).

``` r
compare <- function(theta_hat, theta) {
  p <- pnorm(theta)
  logp <- pnorm(theta, log.p = TRUE)
  q <- pnorm(theta, lower.tail = FALSE)
  logq <- pnorm(theta, log.p = TRUE, lower.tail = FALSE)
  logp_hat <- colMeans(pnorm(theta_hat, log.p = TRUE))
  logq_hat <- colMeans(pnorm(theta_hat, log.p = TRUE, lower.tail = FALSE))
  mean(p * logp + q * logq - p * logp_hat - q * logq_hat)
}
compare(fit_shared$theta_hat_test, thetat)
compare(fit_noshare$theta_hat_test, thetat)
```

The Output
----------

`SharedForest` returns a list containing `mu_hat`, `tau_hat`, and `theta_hat`, which are the sampled values of ![f(X\_i)](https://latex.codecogs.com/png.latex?f%28X_i%29 "f(X_i)"), ![\\sigma^{-2}(X\_i)](https://latex.codecogs.com/png.latex?%5Csigma%5E%7B-2%7D%28X_i%29 "\sigma^{-2}(X_i)"), and ![h(W\_k)](https://latex.codecogs.com/png.latex?h%28W_k%29 "h(W_k)") respectively. Also returned are the values of these quantities on a held-out test set (that you must provide, of course).

The situation is much the same with `GammaForest`, which returns estimated values of ![f(X\_i)](https://latex.codecogs.com/png.latex?f%28X_i%29 "f(X_i)") as `lambda_hat` and ![h(W\_k)](https://latex.codecogs.com/png.latex?h%28W_k%29 "h(W_k)") as `theta_hat`, as well as the shape parameter ![\\alpha](https://latex.codecogs.com/png.latex?%5Calpha "\alpha") as `shape`.

Both functions also return sampled values of the prior variable importances ![s](https://latex.codecogs.com/png.latex?s "s").
