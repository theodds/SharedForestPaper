#include "recbart.h"

using namespace arma;
using namespace Rcpp;

// arma::vec loglik_data(const arma::vec& Y,
//                       const arma::vec& rho,
//                       const Hypers& hypers) {

//   vec out = zeros<vec>(Y.size());
//   for(int i = 0; i < Y.size(); i++) {
//     out(i) = Y(i) * rho(i) - std::exp(rho(i)) - R::lgammafn(Y(i) + 1);
//   }
//   return out;
// }


void IterateGibbsNoS(std::vector<Node*>& forest,
                     MyData& data,
                     const Opts& opts) {

  TreeBackfit(forest, data, opts);
  UpdateShape(forest[0]->hypers, data);
  Rcpp::checkUserInterrupt();
}

void IterateGibbsWithS(std::vector<Node*>& forest,
                       MyData& data,
                       const Opts& opts) {

  IterateGibbsNoS(forest, data, opts);
  if(opts.update_s) UpdateS(forest);
  if(opts.update_alpha) forest[0]->hypers->UpdateAlpha();

}

void TreeBackfit(std::vector<Node*>& forest,
                 MyData& data,
                 const Opts& opts) {

  double MH_BD = 0.7;
  Hypers* hypers = forest[0]->hypers;
  for(int t = 0; t < hypers->num_trees; t++) {
    BackFit(forest[t], data);
    if(forest[t]->is_leaf || unif_rand() < MH_BD) {
      birth_death(forest[t], data);
    }
    else {
      change_decision_rule(forest[t], data);
    }
    forest[t]->UpdateParams(data);
    Refit(forest[t], data);
  }
}

arma::vec predict_reg(Node* tree, MyData& data) {
  int N = data.X.n_rows;
  mat out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = data.X.row(i);
    out.row(i) = predict_reg(tree,x);
  }
  return out;
}

arma::vec predict_reg(Node* tree, arma::mat& X) {
  int N = X.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = predict_reg(tree,x);
  }
  return out;
}

double predict_reg(Node* n, rowvec& x) {
  if(n->is_leaf) {
    return n->lambda;
  }
  if(x(n->var) <= n->val) {
    return predict_reg(n->left, x);
  }
  else {
    return predict_reg(n->right, x);
  }
}

arma::vec predict_theta(std::vector<Node*> forest, arma::mat& W) {
  int N = forest.size();
  vec out = zeros<vec>(W.n_rows);
  for(int n = 0 ; n < N; n++) {
    out = out + predict_theta(forest[n], W);
  }
  return out;
}

arma::vec predict_reg(std::vector<Node*> forest, arma::mat& X) {
  int N = forest.size();
  mat out = zeros<vec>(X.n_rows);
  for(int n = 0 ; n < N; n++) {
    out = out + predict_reg(forest[n], X);
  }
  return out;
}

arma::vec predict_theta(Node* tree, arma::mat& W) {
  int N = W.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec w = W.row(i);
    out(i) = predict_theta(tree,w);
  }
  return out;
}

arma::vec predict_theta(Node* tree, MyData& data) {
  int N = data.W.n_rows;
  vec out = zeros<vec>(N);
  for(int i = 0; i < N; i++) {
    rowvec w = data.W.row(i);
    out(i) = predict_theta(tree,w);
  }
  return out;
}

double predict_theta(Node* n, rowvec& w) {
  if(n->is_leaf) {
    return n->theta;
  }
  if(w(n->var) <= n->val) {
    return predict_theta(n->left, w);
  }
  else {
    return predict_theta(n->right,w);
  }
}

void BackFit(Node* tree, MyData& data) {
  vec lambda = predict_reg(tree, data);
  vec theta = predict_theta(tree, data);
  data.lambda_hat = data.lambda_hat / lambda;
  data.theta_hat = data.theta_hat - theta;
}

void Refit(Node* tree, MyData& data) {
  vec lambda = predict_reg(tree, data);
  vec theta = predict_theta(tree, data);
  data.lambda_hat = data.lambda_hat % lambda;
  data.theta_hat = data.theta_hat + theta;
}

void Node::UpdateParams(MyData& data) {

  UpdateSuffStat(data);
  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();
  double a = pow(hypers->sigma_theta, -2.0);
  for(int i = 0; i < num_leaves; i++) {
    Node* l      = leafs[i];

    double a_up = hypers->a_lambda + l->ss.n * hypers->shape;
    double b_up = hypers->b_lambda + hypers->shape * l->ss.sum_v_Y;

    l->lambda = R::rgamma(a_up, 1.0 / b_up);

    double theta_hat = l->ss.sum_Z / (l->ss.n_Z + a);
    double sigma_theta = pow(l->ss.n_Z + a, -0.5);
    l->theta = theta_hat + norm_rand() * sigma_theta;

  }
}

double Node::LogLT(const MyData& data) {

  UpdateSuffStat(data);
  std::vector<Node*> leafs = leaves(this);


  double out = 0.0;
  int num_leaves = leafs.size();
  double a = pow(hypers->sigma_theta, -2.0);
  double shape = hypers->shape;
  double log_shape = log(hypers->shape);
  double lgamshape = R::lgammafn(shape);
  double a_lambda = hypers->a_lambda;
  double lgama = R::lgammafn(a_lambda);
  double b_lambda = hypers->b_lambda;
  double logb = log(b_lambda);

  for(int i = 0; i < num_leaves; i++) {

    // Define stuff
    Node* l = leafs[i];
    double n = l->ss.n;
    double n_Z = l->ss.n_Z;
    double R_bar = l->ss.sum_Z / n_Z;
    double SSE_Z = l->ss.sum_Z_sq - n_Z * R_bar * R_bar;


    if(n > 0.0) {
      out +=
        n * shape * log_shape
        + l->ss.sum_log_v * shape
        - n * lgamshape
        + (shape - 1.0) * l->ss.sum_log_Y 
        + a_lambda * logb
        - lgama
        + R::lgammafn(a_lambda + n * shape)
        - (a_lambda + n * shape) * log(b_lambda + shape * l->ss.sum_v_Y);
    }

    // Likelihood for classification
    if(n_Z > 0.0) {
      out += 0.5 * log(a / (n_Z + a)) - n_Z * M_LN_SQRT_2PI
        - 0.5 * (SSE_Z + n_Z * a * R_bar * R_bar / (n_Z + a));
    }

  }
  return out;
}

void Node::UpdateSuffStat(const MyData& data) {
  ResetSuffStat();
  int N = data.X.n_rows;
  for(int i = 0; i < N; i++) {
    AddSuffStat(data, i);
  }
  int M = data.W.n_rows;
  for(int i = 0; i < M; i++) {
    AddSuffStatZ(data,i);
  }
}

void Node::AddSuffStat(const MyData& data, int i) {
  ss.sum_log_v += log(data.lambda_hat(i));
  ss.sum_v_Y += data.lambda_hat(i) * data.Y(i);
  ss.sum_log_Y += std::log(data.Y(i));
  ss.n += 1.0;

  if(!is_leaf) {
    double x = data.X(i,var);
    if(x <= val) {
      left->AddSuffStat(data,i);
    } else {
      right->AddSuffStat(data,i);
    }
  }
}

void Node::AddSuffStatZ(const MyData& data, int i) {
  double Z = data.Z(i) - data.theta_hat(i);
  ss.sum_Z += Z;
  ss.sum_Z_sq += Z * Z;
  ss.n_Z += 1.0;

  if(!is_leaf) {
    double w = data.W(i,var);
    if(w <= val) {
      left->AddSuffStatZ(data,i);
    } else {
      right->AddSuffStatZ(data,i);
    }
  }
}

void Node::ResetSuffStat() {
  ss.sum_v_Y    = 0.0;
  ss.sum_log_Y  = 0.0;
  ss.sum_log_v  = 0.0;
  ss.n          = 0.0;
  ss.sum_Z      = 0.0;
  ss.sum_Z_sq   = 0.0;
  ss.n_Z        = 0.0;
  if(!is_leaf) {
    left->ResetSuffStat();
    right->ResetSuffStat();
  }
}

double cauchy_jacobian(double tau, double sigma_hat) {
  double sigma = pow(tau, -0.5);
  int give_log = 1;

  double out = Rf_dcauchy(sigma, 0.0, sigma_hat, give_log);
  out = out - M_LN2 - 3.0 / 2.0 * log(tau);

  return out;

}


// // void UpdateZ(MyData& data) {

// //   int N = data.Z.n_elem;

// //   for(int i = 0; i < N; i++) {

// //     if(data.delta(i) == 0) {
// //       data.Z(i) = randnt(data.theta_hat(i), 1.0, R_NegInf, 0.0);
// //     }
// //     else {
// //       data.Z(i) = randnt(data.theta_hat(i), 1.0, 0.0, R_PosInf);
// //     }
// //   }
// // }

void UpdateShape(Hypers* hypers, MyData& data) {

  double sum_log_v = sum(log(data.lambda_hat));
  double sum_log_Y = sum(log(data.Y));
  double n = data.Y.size();
  double sum_v_Y = sum(data.Y % data.lambda_hat);
  ShapeWeight* shape_weight = new ShapeWeight(sum_log_v, sum_log_Y, sum_v_Y, n);

  double sigma_old = pow(hypers->shape, -0.5);
  double sigma_new = slice_sampler(sigma_old, shape_weight, 1.0, 0.0, R_PosInf);
  hypers->shape = pow(sigma_new, -2.0);
  delete shape_weight;
}

void UpdateZ(MyData& data) {

  int N = data.Z.n_elem;

  for(int i = 0; i < N; i++) {

    double u = unif_rand();
    double Z = 0.0;
    if(data.delta(i) == 0) {
      data.Z(i) = -R::qnorm((1.0 - u) * R::pnorm(data.theta_hat(i), 0.0, 1.0, 1, 0) + u, 0.0, 1.0,1,0);
    }
    else {
      data.Z(i) = R::qnorm((1.0 - u) * R::pnorm(-data.theta_hat(i), 0.0, 1.0,1,0) + u, 0.0, 1.0, 1,0);
    }
    data.Z(i) = data.Z(i) + data.theta_hat(i);
  }
}


// arma::mat get_params(std::vector<Node*>& forest) {
//   std::vector<double> mu(0);
//   std::vector<double> tau(0);
//   std::vector<double> theta(0);
//   int num_tree = forest.size();
//   for(int t = 0; t < num_tree; t++) {
//     get_params(forest[t], mu, tau, theta);
//   }

//   int num_leaves = mu.size();
//   mat mu_tau_theta = zeros<mat>(num_leaves, 3);
//   for(int i = 0; i < num_leaves; i++) {
//     mu_tau_theta(i,0) = mu[0];
//     mu_tau_theta(i,1) = tau[1];
//     mu_tau_theta(i,2) = tau[2];
//   }

//   return mu_tau_theta;

// }

// void get_params(Node* n,
//                 std::vector<double>& mu,
//                 std::vector<double>& tau,
//                 std::vector<double>& theta
//                 )
// {
//   if(n->is_leaf) {
//     mu.push_back(n->mu);
//     tau.push_back(n->tau);
//     theta.push_back(n->theta);
//   }
//   else {
//     get_params(n->left, mu, tau, theta);
//     get_params(n->right, mu, tau, theta);
//   }
// }



