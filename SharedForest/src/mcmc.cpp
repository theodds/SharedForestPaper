#include "recbart.h"

using namespace arma;
using namespace Rcpp;

double probability_node_birth(Node* tree) {
  return tree->is_leaf ? 1.0 : 0.5;
}

Node* birth_node(Node* tree, double* leaf_node_probability) {
  std::vector<Node*> leafs = leaves(tree);
  Node* leaf = rand(leafs);
  *leaf_node_probability = 1.0 / ((double)leafs.size());

  return leaf;
}

Node* death_node(Node* tree, double* p_not_grand) {
  std::vector<Node*> ngb = not_grand_branches(tree);
  Node* branch = rand(ngb);
  *p_not_grand = 1.0 / ((double)ngb.size());

  return branch;
}

void birth_death(Node* tree, const MyData& data) {

  double p_birth = probability_node_birth(tree);

  if(unif_rand() < p_birth) {
    // Rcout << "doing birth\n";
    node_birth(tree, data);
  }
  else {
    // Rcout << "doing death\n";
    node_death(tree, data);
  }
}

void node_birth(Node* tree, const MyData& data) {

  // Rcout << "Sample leaf";
  double leaf_probability = 0.0;
  Node* leaf = birth_node(tree, &leaf_probability);

  // Rcout << "Compute prior";
  // int leaf_depth = leaf->depth;
  int leaf_depth = depth(leaf);
  double leaf_prior = growth_prior(leaf);

  // Get likelihood of current state
  // Rcout << "Current likelihood";
  double ll_before = tree->LogLT(data);
  ll_before += log(1.0 - leaf_prior);

  // Get transition probability
  // Rcout << "Transistion";
  double p_forward = log(probability_node_birth(tree) * leaf_probability);

  // Birth new leaves
  // Rcout << "Birth";
  leaf->BirthLeaves();

  // Get likelihood after
  // Rcout << "New Likelihood";
  double ll_after = tree->LogLT(data);
  ll_after += log(leaf_prior) +
    log(1.0 - growth_prior(leaf->left)) +
    log(1.0 - growth_prior(leaf->right));

  // Get Probability of reverse transition
  // Rcout << "Reverse";
  std::vector<Node*> ngb = not_grand_branches(tree);
  double p_not_grand = 1.0 / ((double)(ngb.size()));
  double p_backward = log((1.0 - probability_node_birth(tree)) * p_not_grand);

  // Do MH
  double log_trans_prob = ll_after + p_backward - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    leaf->DeleteLeaves();
    leaf->var = 0;
    tree->LogLT(data);
  }
  else {
    // Rcout << "Accept!";
  }
}

void node_death(Node* tree, const MyData& data) {

  // Select branch to kill Children
  double p_not_grand = 0.0;
  Node* branch = death_node(tree, &p_not_grand);

  // Compute before likelihood
  // int leaf_depth = branch->left->depth;
  int leaf_depth = depth(branch->left);
  double leaf_prob = growth_prior(branch);
  double left_prior = growth_prior(branch->left);
  double right_prior = growth_prior(branch->right);
  double ll_before = tree->LogLT(data) +
    log(1.0 - left_prior) + log(1.0 - right_prior) + log(leaf_prob);

  // Compute forward transition prob
  double p_forward = log(p_not_grand * (1.0 - probability_node_birth(tree)));

  // Save old leafs, do not delete (they are dangling, need to be handled by the end)
  Node* left = branch->left;
  Node* right = branch->right;
  branch->left = branch;
  branch->right = branch;
  branch->is_leaf = true;

  // Compute likelihood after
  double ll_after = tree->LogLT(data) + log(1.0 - leaf_prob);

  // Compute backwards transition
  std::vector<Node*> leafs = leaves(tree);
  double p_backwards = log(1.0 / ((double)(leafs.size())) * probability_node_birth(tree));

  // Do MH and fix dangles
  double log_trans_prob = ll_after + p_backwards - ll_before - p_forward;
  if(log(unif_rand()) > log_trans_prob) {
    branch->left = left;
    branch->right = right;
    branch->is_leaf = false;
    tree->LogLT(data);
  }
  else {
    delete left;
    delete right;
  }
}

void change_decision_rule(Node* tree,
                          const MyData& data) {

  std::vector<Node*> ngb = not_grand_branches(tree);
  Node* branch = rand(ngb);

  // Calculate likelihood before proposal
  double ll_before = tree->LogLT(data);

  // save old split
  int old_feature = branch->var;
  double old_value = branch->val;
  double old_lower = branch->lower;
  double old_upper = branch->upper;

  // Modify the branch
  // branch->var = sample_class(hypers.s);
  branch->var = tree->hypers->SampleVar();
  branch->GetLimits();
  branch->val = (branch->upper - branch->lower) * unif_rand() + branch->lower;

  // Calculate likelihood after proposal
  double ll_after = tree->LogLT(data);

  // Do MH
  double log_trans_prob = ll_after - ll_before;

  if(log(unif_rand()) > log_trans_prob) {
    branch->var = old_feature;
    branch->val = old_value;
    branch->lower = old_lower;
    branch->upper = old_upper;
    tree->LogLT(data);
  }
}

/*Note: Because the shape of the Dirichlet will mostly be small, we sample from
  the Dirichlet distribution by sampling log-gamma random variables using the
  technique of Liu, Martin, and Syring (2017+) and normalizing using the
  log-sum-exp trick */
void UpdateS(std::vector<Node*>& forest) {

  Hypers* hypers = forest[1]->hypers;

  // Get shape vector
  vec shape_up = hypers->alpha / ((double)hypers->s.size()) * ones<vec>(hypers->s.size());
  shape_up = shape_up + get_var_counts(forest, *hypers);

  // Sample unnormalized s on the log scale
  for(int i = 0; i < shape_up.size(); i++) {
    hypers->logs(i) = rlgam(shape_up(i));
  }
  // Normalize s on the log scale, then exponentiate
  hypers->logs = hypers->logs - log_sum_exp(hypers->logs);
  hypers->s = exp(hypers->logs);

}

void do_burn_in(std::vector<Node*> forest,
                MyData& data,
                const Opts& opts) {

  for(int i = 0; i < opts.num_burn; i++) {
    if(i < opts.num_burn / 2) {
      IterateGibbsNoS(forest, data, opts);
    }
    else {
      IterateGibbsWithS(forest, data, opts);
    }

    if((i+1) % opts.num_print == 0) {
      Rcpp::Rcout << "Finishing warmup " << i + 1 << "\n";
    }
  }
}
