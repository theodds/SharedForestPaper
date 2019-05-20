// void Frailty::UpdateFrailty(arma::vec& rho) {

//   arma::vec R = zeros<vec>(num_clusters);

//   for(int i = 0; i < num_clusters; i++) {

//     // Get residual and add to R
//     for(int j = 0; j < cluster_to_users[i].size(); j++) {
//       int istar = cluster_to_users[i][j];
//       rho(istar) = rho(istar) - frailty(i);
//       R(i) = R(i) + std::exp(rho(istar));
//     }

//     // Update
//     frailty(i) = log(Rf_rgamma(shape + N(i), 1.0 / (shape + R(i))));

//     // Fix rho
//     for(int j = 0; j < cluster_to_users[i].size(); j++) {
//       int istar = cluster_to_users[i][j];
//       rho(istar) = rho(istar) + frailty(i);
//     }
//   }

// }

// void Frailty::UpdateMu0(arma::vec& rho) {

//   double R = 0.0;

//   // Get residual and add to R
//   for(int i = 0; i < rho.size(); i++) {
//     rho(i) = rho(i) - mu_0;
//     R = R + std::exp(rho(i));
//   }

//   // Update
//   mu_0 = log(Rf_rgamma(N_tot, 1.0 / R));

//   // Fix rho
//   for(int i = 0; i < rho.size(); i++) {
//     rho(i) = rho(i) + mu_0;
//   }
// }

// void Frailty::UpdateShape() {

//   double sum_frailty = sum(frailty);
//   double sum_exp_frailty = sum(exp(frailty));
//   double num_clust = (double)(num_clusters);

//   ShapeLoglik* loglik = new ShapeLoglik(sum_frailty, sum_exp_frailty, num_clust);

//   double scale_current = 1.0 / sqrt(shape);
//   double scale_up = slice_sampler(scale_current, loglik, 1.0, 0.0, R_PosInf);

//   shape = 1.0 / (scale_up * scale_up);

//   delete loglik;

// }


// List PoisBart(const arma::mat& X,
//               const arma::vec& Y,
//               const arma::mat& X_test,
//               const arma::uvec& group,
//               double alpha,
//               double beta,
//               double gamma,
//               int num_trees,
//               double a_nu,
//               double b_nu,
//               double alpha_scale,
//               double alpha_shape_1,
//               double alpha_shape_2,
//               bool update_s,
//               bool update_alpha,
//               int num_burn,
//               int num_thin,
//               int num_save,
//               int num_print) {

//   Opts opts = Opts(num_burn, num_thin, num_save, num_print,
//                    update_s, update_alpha);

//   Hypers hypers = Hypers(X, group, alpha, beta, gamma, a_nu, b_nu, num_trees,
//                          alpha_scale, alpha_shape_1, alpha_shape_2);

//   return do_pois_bart(X,Y,X_test,hypers,opts);

// }

// arma::vec predict(const std::vector<Node*>& forest,
//                   const arma::mat& X,
//                   const Hypers& hypers) {

//   vec out = zeros<vec>(X.n_rows);
//   int num_tree = forest.size();

//   for(int t = 0; t < num_tree; t++) {
//     out = out + predict(forest[t], X, hypers);
//   }

//   return out;
// }


// arma::vec predict(Node* n, const arma::mat& X, const Hypers& hypers) {

//   int N = X.n_rows;
//   vec out = zeros<vec>(N);

//   for(int i = 0; i < N; i++) {
//     rowvec x = X.row(i);
//     out(i) = predict(n, x);
//   }

//   return out;
// }

// double predict(Node* n, const arma::rowvec& x) {

//   if(n->is_leaf) {
//     return n->mu;
//   }
//   if(x(n->var) <= n->val) {
//     return predict(n->left, x);
//   }
//   else {
//     return predict(n->right, x);
//   }
// }

// List do_pois_bart(const arma::mat& X, const arma::vec& Y,
//                   const arma::mat& X_test, Hypers& hypers, const Opts& opts) {


//   // Make Forest
//   std::vector<Node*> forest = init_forest(X, Y, hypers);

//   // Make rho
//   vec rho = zeros<vec>(X.n_rows);

//   // Do burnin
//   do_burn_in(forest, rho, hypers, X, Y, opts);

//   // Make Save state structure
//   SaveState savestate = SaveState(opts, X, X_test, hypers);

//   // Do Gibbs iterates
//   do_gibbs_iterates(forest, rho, hypers, X, Y, X_test, opts, savestate);

//   // Return output
//   List out;
//   out["eta_hat_train"] = savestate.eta_hat_train;
//   out["eta_hat_test"] = savestate.eta_hat_test;
//   out["alpha"] = savestate.alpha;
//   out["s"] = savestate.s;
//   out["var_counts"] = savestate.var_counts;
//   out["loglik"] = savestate.loglik;
//   out["loglik_train"] = savestate.loglik_train;

//   return out;

// }
