#include "recbart.h"

using namespace arma;
using namespace Rcpp;

void Hypers::UpdateAlpha() {

  RhoLoglik* loglik = new RhoLoglik(mean(logs), (double)s.size(), alpha_scale,
                                    alpha_shape_1, alpha_shape_2);

  double rho_current = alpha / (alpha + alpha_scale);
  double rho_up = slice_sampler(rho_current, loglik, 0.1, 0.0, 1.0);

  alpha = alpha_scale * rho_up / (1.0 - rho_up);

  delete loglik;

}

int Hypers::SampleVar() const {

  int group_idx = sample_class(s);
  int var_idx = sample_class(group_to_vars[group_idx].size());

  return group_to_vars[group_idx][var_idx];
}

Hypers::Hypers(const mat& X,
               const arma::uvec& group,
               Rcpp::List hypers) {

  alpha         = hypers["alpha"];
  beta          = hypers["beta"];
  gamma         = hypers["gamma"];
  num_trees     = hypers["num_tree"];
  a_lambda      = hypers["a_lambda"];
  b_lambda      = hypers["b_lambda"];
  alpha_scale   = hypers["alpha_scale"];
  alpha_shape_1 = hypers["alpha_shape_1"];
  alpha_shape_2 = hypers["alpha_shape_2"];
  sigma_theta   = hypers["sigma_theta"];
  theta_0       = hypers["theta_0"];
  lambda_0      = hypers["lambda_0"];
  shape         = hypers["shape"];
  num_groups    = group.max() + 1;
  s             = ones<vec>(num_groups)/((double)num_groups);
  logs          = log(s);
  this->group   = group;


  group_to_vars.resize(s.size());
  for(int i = 0; i < s.size(); i++) {
    group_to_vars[i].resize(0);
  }
  int P = group.size();
  for(int p = 0; p < P; p++) {
    int idx = group(p);
    group_to_vars[idx].push_back(p);
  }
}
