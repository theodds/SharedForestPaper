#include "recbart.h"

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List SharedBart(arma::mat& X,
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

  MyData data(X,W,Y,delta,hypers.tau_0, hypers.theta_0);

  mat mu_hat = zeros<mat>(opts.num_save, X.n_rows);
  mat tau_hat = zeros<mat>(opts.num_save, X.n_rows);
  mat theta_hat = zeros<mat>(opts.num_save, W.n_rows);
  mat s = zeros<mat>(opts.num_save, hypers.num_groups);
  mat mu_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
  mat tau_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
  mat theta_hat_test = zeros<mat>(opts.num_save, W_test.n_rows);

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
    mu_hat.row(i) = trans(data.mu_hat);
    tau_hat.row(i) = trans(data.tau_hat);
    theta_hat.row(i) = trans(data.theta_hat);
    s.row(i) = trans(hypers.s);
    mat mutau = predict_reg(forest, X_test);
    mu_hat_test.row(i) = trans(mutau.col(0));
    tau_hat_test.row(i) = trans(mutau.col(1)) * hypers.tau_0;
    theta_hat_test.row(i) = trans(predict_theta(forest, W_test)) + hypers.theta_0;
  }
  Rcout << std::endl;

  Rcout << "Number of leaves at final iterations:\n";
  for(int t = 0; t < hypers.num_trees; t++) {
    Rcout << leaves(forest[t]).size() << " ";
    if((t + 1) % 10 == 0) Rcout << "\n";
  }

  List out;
  out["mu_hat"] = mu_hat;
  out["tau_hat"] = tau_hat;
  out["theta_hat"] = theta_hat;
  out["mu_hat_mean"] = mean(mu_hat, 0);
  out["theta_hat_mean"] = mean(theta_hat, 0);
  out["s"] = s;
  out["mu_hat_test"] = mu_hat_test;
  out["tau_hat_test"] = tau_hat_test;
  out["theta_hat_test"] = theta_hat_test;

  return out;
}

// [[Rcpp::export]]
List MixedBart(arma::mat& X,
               arma::vec& Y,
               arma::mat& W,
               arma::uvec& delta,
               arma::uvec& cluster,
               arma::uvec& clusterw,
               List hypers_,
               List opts_) {

  arma::uvec group = hypers_["group"];
  Hypers hypers(X, group, hypers_);
  Opts opts(opts_);
  MyData data(X,W,Y,delta,hypers.tau_0, hypers.theta_0);
  Cluster my_cluster(cluster-1, clusterw-1);

  mat mu_hat = zeros<mat>(opts.num_save, X.n_rows);
  mat tau_hat = zeros<mat>(opts.num_save, X.n_rows);
  mat theta_hat = zeros<mat>(opts.num_save, W.n_rows);
  mat s = zeros<mat>(opts.num_save, X.n_cols);
  // mat mu_hat_test = zeros<mat>(opts.num_save, X_test.n_rows);
  // mat theta_hat_test = zeros<mat>(opts.num_save, W_test.n_rows);
  vec sigma_V_mu = zeros<vec>(opts.num_save);
  vec sigma_V_theta = zeros<vec>(opts.num_save);
  vec a_V_tau = zeros<vec>(opts.num_save);
  mat V_mu = zeros<mat>(opts.num_save, my_cluster.V_mu.size());
  mat V_theta = zeros<mat>(opts.num_save, my_cluster.V_theta.size());
  mat V_tau = zeros<mat>(opts.num_save, my_cluster.V_tau.size());

  std::vector<Node*> forest = init_forest(hypers);

  for(int i = 0; i < opts.num_burn; i++) {
    if(i > opts.num_burn / 2) {
      IterateGibbsWithS(forest, data, opts);
    }
    else {
      IterateGibbsNoS(forest, data, opts);
    }
    UpdateZ(data);
    UpdateVmu(my_cluster, data);
    UpdateVtheta(my_cluster, data);
    UpdateVars(my_cluster, data);
    UpdateVtau(my_cluster, data);
    UpdateShape(my_cluster);
    if(i % opts.num_print == 0) Rcout << "Finishing warmup " << i << "\t\t\r";
    // if(i % 100 == 0) Rcout << "Finishing warmup " << i << std::endl;
  }

  Rcout << std::endl;

  for(int i = 0; i < opts.num_save; i++) {
    for(int j = 0; j < opts.num_thin; j++) {
      IterateGibbsWithS(forest, data, opts);
      UpdateZ(data);
      UpdateVmu(my_cluster, data);
      UpdateVtheta(my_cluster, data);
      UpdateVars(my_cluster, data);
      UpdateVtau(my_cluster, data);
      UpdateShape(my_cluster);
    }
    if(i % opts.num_print == 0) Rcout << "Finishing save " << i << "\t\t\r";
    // if(i % 100 == 0) Rcout << "Finishing save " << i << std::endl;
    mu_hat.row(i) = trans(data.mu_hat);
    tau_hat.row(i) = trans(data.tau_hat);
    theta_hat.row(i) = trans(data.theta_hat);
    s.row(i) = trans(hypers.s);
    // mat mutau = predict_reg(forest, X_test);
    // mu_hat_test.row(i) = trans(mutau.col(0));
    // theta_hat_test.row(i) = trans(predict_theta(forest, W_test)) + hypers.theta_0;
    sigma_V_mu(i) = my_cluster.sigma_V_mu;
    sigma_V_theta(i) = my_cluster.sigma_V_theta;
    V_mu.row(i) = trans(my_cluster.V_mu);
    V_theta.row(i) = trans(my_cluster.V_theta);
    V_tau.row(i) = trans(my_cluster.V_tau);
    a_V_tau(i) = my_cluster.a_V_tau;
  }
  Rcout << std::endl;

  Rcout << "Number of leaves at final iterations:\n";
  for(int t = 0; t < hypers.num_trees; t++) {
    Rcout << leaves(forest[t]).size() << " ";
    if((t + 1) % 10 == 0) Rcout << "\n";
  }

  List out;
  out["mu_hat"] = mu_hat;
  out["tau_hat"] = tau_hat;
  out["theta_hat"] = theta_hat;
  out["mu_hat_mean"] = mean(mu_hat, 0);
  out["theta_hat_mean"] = mean(theta_hat, 0);
  out["s"] = s;
  // out["mu_hat_test"] = mu_hat_test;
  // out["theta_hat_test"] = theta_hat_test;
  out["sigma_V_mu"] = sigma_V_mu;
  out["sigma_V_theta"] = sigma_V_theta;
  out["V_mu"] = V_mu;
  out["V_theta"] = V_theta;
  out["V_tau"] = V_tau;
  out["a_V_tau"] = a_V_tau;

  return out;
}

// // [[Rcpp::export]]
// List HBart(const arma::mat& X,
//            const arma::vec& Y,
//            List hypers_,
//            List opts_) {

//   arma::uvec group = hypers_["group"];
//   Hypers hypers(X, group, hypers_);
//   Opts opts(opts_);

// }

// // [[Rcpp::export]]
// List RegBart(arma::mat X,
//              arma::vec Y,
//              List hypers_,
//              List opts_) {
//   arma::uvec group = hypers_["group"];
//   Hypers hypers(X, group, hypers_);
//   Opts opts(opts_);

//   uvec delta = zeros<uvec>(Y.size());
//   MyData data(X, X, Y, delta, hypers.tau_0, hypers.theta_0);

//   mat mu_hat = zeros<mat>(opts.num_burn, X.n_rows);
//   mat tau_hat = zeros<mat>(opts.num_burn, X.n_rows);
//   vec tau_0  = zeros<vec>(opts.num_burn);

//   std::vector<Node*> forest = init_forest(hypers);

//   for(int i = 0; i < opts.num_burn; i++) {
//     IterateGibbsNoS(forest, data, opts);
//     mu_hat.row(i) = trans(data.mu_hat);
//     tau_hat.row(i) = trans(data.tau_hat);
//     tau_0(i) = forest[0]->hypers->tau_0;
//     if(i % 100 == 0) Rcout << "Finishing warmup " << i << std::endl;
//   }

//   Rcout << "Number of leaves at final iterations:\n";
//   for(int t = 0; t < hypers.num_trees; t++) {
//     Rcout << leaves(forest[t]).size() << " ";
//     if((t + 1) % 10 == 0) Rcout << "\n";
//   }

//   List out;
//   out["mu_hat"] = mu_hat;
//   out["tau_hat"] = tau_hat;
//   out["tau_0"] = tau_0;

//   return out;

// }
