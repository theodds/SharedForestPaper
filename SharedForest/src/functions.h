#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <RcppArmadillo.h>

int sample_class(const arma::vec& probs);

int sample_class(int n);

double logit(double x);

double expit(double x);

double log_sum_exp(const arma::vec& x);


double rlgam(double shape);

arma::vec rdirichlet(const arma::vec& shape);

double alpha_to_rho(double alpha, double scale);

double rho_to_alpha(double rho, double scale);

double logpdf_beta(double x, double a, double b);

bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new);

double randnt(double lower, double upper);

double randnt(double mu, double sigma, double lower, double upper);
#endif
