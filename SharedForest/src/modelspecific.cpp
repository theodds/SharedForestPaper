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
  forest[0]->hypers->UpdateTau(data);
  // UpdateSigmaParam(forest);

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

arma::mat predict_reg(Node* tree, MyData& data) {
  int N = data.X.n_rows;
  mat out = zeros<mat>(N, 2);
  for(int i = 0; i < N; i++) {
    rowvec x = data.X.row(i);
    out.row(i) = predict_reg(tree,x);
  }
  return out;
}

arma::mat predict_reg(Node* tree, arma::mat& X) {
  int N = X.n_rows;
  mat out = zeros<mat>(N, 2);
  for(int i = 0; i < N; i++) {
    rowvec x = X.row(i);
    out.row(i) = predict_reg(tree,x);
  }
  return out;
}

arma::rowvec predict_reg(Node* n, rowvec& x) {
  if(n->is_leaf) {
    rowvec out = zeros<rowvec>(2);
    out(0) = n->mu;
    out(1) = n->tau;
    return out;
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

arma::mat predict_reg(std::vector<Node*> forest, arma::mat& X) {
  int N = forest.size();
  mat out = zeros<mat>(X.n_rows,2);
  out.col(1) = ones<vec>(X.n_rows);
  for(int n = 0 ; n < N; n++) {
    mat mutau = predict_reg(forest[n], X);
    out.col(0) = out.col(0) + mutau.col(0);
    out.col(1) = out.col(1) % mutau.col(1);
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
  mat mu_tau = predict_reg(tree, data);
  vec theta = predict_theta(tree, data);
  data.mu_hat = data.mu_hat - mu_tau.col(0);
  data.tau_hat = data.tau_hat / mu_tau.col(1);
  data.theta_hat = data.theta_hat - theta;
}

void Refit(Node* tree, MyData& data) {
  mat mu_tau = predict_reg(tree, data);
  vec theta = predict_theta(tree, data);
  data.mu_hat = data.mu_hat + mu_tau.col(0);
  data.tau_hat = data.tau_hat % mu_tau.col(1);
  data.theta_hat = data.theta_hat + theta;
}

void Node::UpdateParams(MyData& data) {

  UpdateSuffStat(data);
  std::vector<Node*> leafs = leaves(this);
  int num_leaves = leafs.size();
  double a = 1.0/pow(hypers->sigma_theta,2);
  for(int i = 0; i < num_leaves; i++) {
    Node* l      = leafs[i];
    double w     = l->ss.sum_v;
    double Y_bar = l->ss.sum_v_Y / l->ss.sum_v;
    double SSE   = l->ss.sum_v_Y_sq - l->ss.sum_v * Y_bar * Y_bar;
    double kappa = hypers->kappa;

    double mu_up = l->ss.sum_v_Y / (w + kappa);
    double kappa_up = kappa + w;
    double a_up = hypers->a_tau + 0.5 * l->ss.n;
    double b_up = hypers->b_tau + 0.5 * SSE
      + 0.5 * kappa * w * Y_bar * Y_bar / (kappa + w);

    l->tau = R::rgamma(a_up, 1.0 / b_up);
    l->mu  = mu_up + norm_rand() / sqrt(l->tau * kappa_up);

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
  double kappa = hypers->kappa;
  double a_tau = hypers->a_tau;
  double b_tau = hypers->b_tau;
  double a     = 1.0 / (hypers->sigma_theta * hypers->sigma_theta);

  for(int i = 0; i < num_leaves; i++) {

    // Define stuff
    Node* l = leafs[i];
    double w = l->ss.sum_v;
    double Y_bar = l->ss.sum_v_Y / l->ss.sum_v;
    double SSE = l->ss.sum_v_Y_sq - Y_bar * Y_bar * l->ss.sum_v;
    double sum_log_tau = l->ss.sum_log_v;
    double n = l->ss.n;
    double n_Z = l->ss.n_Z;
    double R_bar = l->ss.sum_Z / n_Z;
    double SSE_Z = l->ss.sum_Z_sq - n_Z * R_bar * R_bar;

    // Likelihood for regression
    if(n > 0.0) {
      out += 0.5 * sum_log_tau - n * M_LN_SQRT_2PI
        + 0.5 * std::log(kappa / (kappa + w)) + R::lgammafn(a_tau + 0.5 * n)
        - (a_tau + 0.5 * n)
        * std::log(b_tau + 0.5*SSE + 0.5*kappa*w*Y_bar*Y_bar / (kappa + w))
        + a_tau * log(b_tau) - R::lgammafn(a_tau);
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
  double Z = data.Y(i) - data.mu_hat(i);
  ss.sum_v   += data.tau_hat(i);
  ss.sum_v_Y += Z * data.tau_hat(i);
  ss.sum_v_Y_sq += Z * Z * data.tau_hat(i);
  ss.sum_log_v += std::log(data.tau_hat(i));
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
  ss.sum_v_Y_sq = 0.0;
  ss.sum_v      = 0.0;
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

void Hypers::UpdateTau(MyData& data) {

  arma::vec res = data.Y - data.mu_hat;
  data.tau_hat = data.tau_hat / tau_0;

  double SSE = sum(data.tau_hat % res % res);
  double n = res.size();

  double shape = 0.5 * n + 1.0;
  double scale = 2.0 / SSE;
  double sigma_prop = pow(Rf_rgamma(shape, scale), -0.5);
  double sigma_old = pow(tau_0, -0.5);

  double tau_prop = pow(sigma_prop, -2.0);

  double loglik_rat = cauchy_jacobian(tau_prop, scale_sigma) -
    cauchy_jacobian(tau_0, scale_sigma);

  tau_0 = log(unif_rand()) < loglik_rat ? tau_prop : tau_0;

  // SigmaLoglik* loglik = new SigmaLoglik(n, SSE, scale_sigma);
  // double sigma_0 = pow(tau_0, -0.5);
  // sigma_0 = slice_sampler(sigma_0, loglik, 1.0, 0.0, 1000.0);
  // tau_0 = pow(sigma_0, -2.0);

  data.tau_hat = data.tau_hat * tau_0;

}

// void UpdateZ(MyData& data) {

//   int N = data.Z.n_elem;

//   for(int i = 0; i < N; i++) {

//     if(data.delta(i) == 0) {
//       data.Z(i) = randnt(data.theta_hat(i), 1.0, R_NegInf, 0.0);
//     }
//     else {
//       data.Z(i) = randnt(data.theta_hat(i), 1.0, 0.0, R_PosInf);
//     }
//   }
// }


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

void UpdateSigmaParam(std::vector<Node*>& forest) {

  mat mu_tau_theta = get_params(forest);
  vec mu = mu_tau_theta.col(0);
  vec tau = mu_tau_theta.col(1);
  vec theta = mu_tau_theta.col(2);

  double Lambda_kappa = sum(mu % mu % tau);
  double Lambda_theta = sum(theta % theta);
  double num_leaves = mu.size();
  double sigma_theta_hat = forest[0]->hypers->sigma_theta_hat;
  double sigma_mu_hat = forest[0]->hypers->sigma_mu_hat;

  // Update kappa
  double kappa_old = forest[0]->hypers->kappa;
  double kappa_new = Rf_rgamma(1.0 + 0.5 * num_leaves, 1.0 / (0.5 * Lambda_kappa));
  double loglik_rat = cauchy_jacobian(kappa_new, sigma_mu_hat) - cauchy_jacobian(kappa_old, sigma_mu_hat);

  forest[0]->hypers->kappa = log(unif_rand()) < loglik_rat ? kappa_new : kappa_old;

  // Update sigma_theta
  double prec_theta_old = pow(forest[0]->hypers->sigma_theta, -2.0);
  double prec_theta_new = Rf_rgamma(1.0 + 0.5 * num_leaves, 1.0 / (0.5 * Lambda_theta));
  loglik_rat = cauchy_jacobian(prec_theta_new, sigma_theta_hat) - cauchy_jacobian(prec_theta_old, sigma_theta_hat);
  double prec = log(unif_rand()) < loglik_rat ? prec_theta_new : prec_theta_old;

  forest[0]->hypers->sigma_theta = pow(prec, -0.5);

}


arma::mat get_params(std::vector<Node*>& forest) {
  std::vector<double> mu(0);
  std::vector<double> tau(0);
  std::vector<double> theta(0);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_params(forest[t], mu, tau, theta);
  }

  int num_leaves = mu.size();
  mat mu_tau_theta = zeros<mat>(num_leaves, 3);
  for(int i = 0; i < num_leaves; i++) {
    mu_tau_theta(i,0) = mu[i];
    mu_tau_theta(i,1) = tau[i];
    mu_tau_theta(i,2) = theta[i];
  }

  return mu_tau_theta;

}

void get_params(Node* n,
                std::vector<double>& mu,
                std::vector<double>& tau,
                std::vector<double>& theta
                )
{
  if(n->is_leaf) {
    mu.push_back(n->mu);
    tau.push_back(n->tau);
    theta.push_back(n->theta);
  }
  else {
    get_params(n->left, mu, tau, theta);
    get_params(n->right, mu, tau, theta);
  }
}

Cluster::Cluster(const arma::uvec& cluster, const arma::uvec& clusterw) {

  this->cluster = cluster;
  this->clusterw = clusterw;

  int num_cluster = cluster.max() + 1;
  int num_clusterw = clusterw.max() + 1;
  V_mu = zeros<vec>(num_cluster);
  V_theta = zeros<vec>(num_clusterw);
  V_tau = ones<vec>(num_cluster);

  sigma_V_mu    = 0.2 * exp_rand();
  sigma_V_theta = 0.2 * exp_rand();
  a_V_tau       = 100.0;

  cluster_to_idx.resize(num_cluster);
  for(int i = 0; i < num_cluster; i++) {
    cluster_to_idx[i] = find(cluster == i);
  }
  clusterw_to_idx.resize(num_clusterw);
  for(int i = 0; i < num_clusterw; i++) {
    clusterw_to_idx[i] = find(clusterw == i);
  }
}

void UpdateVmu(Cluster& cluster, MyData& data) {

  int nclust = cluster.V_mu.size();
  double a = pow(cluster.sigma_V_mu, -2.0);

  for(int i = 0; i < nclust; i++) {

    double sum_tau = 0.0;
    double sum_tau_R = 0.0;
    int clust_size = cluster.cluster_to_idx[i].size();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      sum_tau += data.tau_hat(n);
      data.mu_hat(n) = data.mu_hat(n) - cluster.V_mu(i);
      sum_tau_R += data.tau_hat(n) * (data.Y(n) - data.mu_hat(n));
    }

    double mu_up = sum_tau_R / (sum_tau + a);
    double sigma_up = pow(sum_tau + a, -0.5);
    cluster.V_mu(i) = mu_up + sigma_up * norm_rand();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      data.mu_hat(n) = data.mu_hat(n) + cluster.V_mu(i);
    }
  }
}

void UpdateVtheta(Cluster& cluster, MyData& data) {

  int nclust = cluster.V_theta.size();
  double a = pow(cluster.sigma_V_theta, -2.0);

  for(int i = 0; i < nclust; i++) {
    double sum_tau = 0.0;
    double sum_R = 0.0;
    int clust_size = cluster.clusterw_to_idx[i].size();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.clusterw_to_idx[i](j);
      sum_tau += 1;
      data.theta_hat(n) = data.theta_hat(n) - cluster.V_theta(i);
      sum_R += data.Z(n) - data.theta_hat(n);
    }

    double mu_up = sum_R / (sum_tau + a);
    double sigma_up = pow(sum_tau + a, -0.5);
    cluster.V_theta(i) = mu_up + sigma_up * norm_rand();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.clusterw_to_idx[i](j);
      data.theta_hat(n) = data.theta_hat(n) + cluster.V_theta(i);
    }
  }
}

void UpdateVtau(Cluster& cluster, MyData& data) {
  int nclust = cluster.V_tau.size();

  for(int i = 0; i < nclust; i++) {
    double ns = 0.0;
    double SS = 0.0;
    int clust_size = cluster.cluster_to_idx[i].size();

    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      ns += 1.0;
      double a = data.tau_hat(n);
      double b = cluster.V_tau(i);
      double c = a + b;
      data.tau_hat(n) = data.tau_hat(n) / cluster.V_tau(i);
      SS += data.tau_hat(n) * pow(data.Y(n) - data.mu_hat(n), 2.0);
    }
    cluster.V_tau(i) = R::rgamma(cluster.a_V_tau + 0.5 * ns, 1.0 / (cluster.a_V_tau + 0.5 * SS));
    for(int j = 0; j < clust_size; j++) {
      int n = cluster.cluster_to_idx[i](j);
      data.tau_hat(n) = data.tau_hat(n) * cluster.V_tau(i);
    }
  }
}

void UpdateShape(Cluster& cluster) {
  ShapeLoglik shape(sum(log(cluster.V_tau)),
                    sum(cluster.V_tau),
                    cluster.V_tau.size());

  double sigma = 1.0 / sqrt(cluster.a_V_tau);
  cluster.a_V_tau = pow(slice_sampler(sigma, &shape, 1.0, 0.0, 10000.0), -2.0);

}
void UpdateVars(Cluster& cluster, MyData& data) {

  double N_mu = cluster.V_mu.size();
  double N_theta = cluster.V_theta.size();
  double SSE_mu = (N_mu - 1.0) * var(cluster.V_mu);
  double SSE_theta = (N_theta - 1.0) * var(cluster.V_theta);

  double tau_mu = R::rgamma(0.5 * N_mu, 2.0 / SSE_mu);
  double tau_theta = R::rgamma(0.5 * N_theta, 2.0 / SSE_theta);

  cluster.sigma_V_mu = pow(tau_mu, -0.5);
  cluster.sigma_V_theta = pow(tau_theta, -0.5);

}

