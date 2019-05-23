#include "recbart.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List GammaBart(arma::mat& X,
               arma::vec& Y,
               arma::mat& W,
               arma::uvec& delta,
               arma::mat& X_test,
               arma::mat& W_test,
               List hypers_,
               List opts_) {
  arma::uvec group = hypers_["group"];
  Hypers hypers(X, group, hypers_);
  Opts opts(opts_);

  MyData data(X,W,Y,delta,hypers.lambda_0, hypers.theta_0);

  mat lambda_hat = zeros<mat>(opts.num_save, X.n_rows);
  mat theta_hat = zeros<mat>(opts.num_save, W.n_rows);
  mat s = zeros<mat>(opts.num_save, X.n_cols);
  mat lambda_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
  mat theta_hat_test = zeros<mat>(opts.num_save, W_test.n_rows);
  vec shape = zeros<vec>(opts.num_save);

  std::vector<Node*> forest = init_forest(hypers);

  for(int i = 0; i < opts.num_burn; i++) {
    if(i > opts.num_burn / 2) {
      IterateGibbsWithS(forest, data, opts);
    }
    else {
      IterateGibbsNoS(forest, data, opts);
    }
    UpdateZ(data);
    if(i % opts.num_print == 0) Rcout << "Finishing warmup " << i << "\t\t\r";
    // if(i % 100 == 0) Rcout << "Finishing warmup " << i << std::endl;
  }

  Rcout << std::endl;

  for(int i = 0; i < opts.num_save; i++) {
    for(int j = 0; j < opts.num_thin; j++) {
      IterateGibbsWithS(forest, data, opts);
      UpdateZ(data);
    }
    if(i % opts.num_print == 0) Rcout << "Finishing save " << i << "\t\t\r";
    // if(i % 100 == 0) Rcout << "Finishing save " << i << std::endl;
    lambda_hat.row(i) = trans(data.lambda_hat);
    theta_hat.row(i) = trans(data.theta_hat);
    s.row(i) = trans(hypers.s);
    lambda_hat_test.row(i) = trans(predict_reg(forest, X_test)) * hypers.lambda_0;
    theta_hat_test.row(i) = trans(predict_theta(forest, W_test)) + hypers.theta_0;
    shape(i) = forest[0]->hypers->shape;
  }
  Rcout << std::endl;

  Rcout << "Number of leaves at final iterations:\n";
  for(int t = 0; t < hypers.num_trees; t++) {
    Rcout << leaves(forest[t]).size() << " ";
    if((t + 1) % 10 == 0) Rcout << "\n";
  }

  List out;
  out["lambda_hat"] = lambda_hat;
  out["theta_hat"] = theta_hat;
  out["lambda_hat_mean"] = mean(lambda_hat, 0);
  out["theta_hat_mean"] = mean(theta_hat, 0);
  out["s"] = s;
  out["lambda_hat_test"] = lambda_hat_test;
  out["theta_hat_test"] = theta_hat_test;
  out["shape"] = shape;

  return out;
}


