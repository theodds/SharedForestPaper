#include "recbart.h"

using namespace arma;
using namespace Rcpp;

// double growth_prior(Node* node) {
//   return node->hypers->gamma * std::pow(1.0 + node->depth, -node->hypers->beta);
// }


double growth_prior(Node* node) {
  return node->hypers->gamma * std::pow(1.0 + depth(node), -node->hypers->beta);
}

int depth(Node* node) {
  return node->is_root ? 0 : 1 + depth(node->parent);
}

void leaves(Node* x, std::vector<Node*>& leafs) {
  if(x->is_leaf) {
    leafs.push_back(x);
  }
  else {
    leaves(x->left, leafs);
    leaves(x->right, leafs);
  }
}

std::vector<Node*> leaves(Node* x) {
  std::vector<Node*> leafs(0);
  leaves(x, leafs);
  return leafs;
}

std::vector<Node*> not_grand_branches(Node* tree) {
  std::vector<Node*> ngb(0);
  not_grand_branches(ngb, tree);
  return ngb;
}

void not_grand_branches(std::vector<Node*>& ngb, Node* node) {
  if(!node->is_leaf) {
    bool left_is_leaf = node->left->is_leaf;
    bool right_is_leaf = node->right->is_leaf;
    if(left_is_leaf && right_is_leaf) {
      ngb.push_back(node);
    }
    else {
      not_grand_branches(ngb, node->left);
      not_grand_branches(ngb, node->right);
    }
  }
}

Node* rand(std::vector<Node*> ngb) {

  int N = ngb.size();
  arma::vec p = ones<vec>(N) / ((double)(N));
  int i = sample_class(p);
  return ngb[i];
}


arma::uvec get_var_counts(std::vector<Node*>& forest, const Hypers& hypers) {
  arma::uvec counts = zeros<uvec>(hypers.num_groups);
  int num_tree = forest.size();
  for(int t = 0; t < num_tree; t++) {
    get_var_counts(counts, forest[t], hypers);
  }
  return counts;
}

void get_var_counts(arma::uvec& counts, Node* node, const Hypers& hypers) {
  if(!node->is_leaf) {
    int group_idx = hypers.group(node->var);
    counts(group_idx) = counts(group_idx) + 1;
    get_var_counts(counts, node->left, hypers);
    get_var_counts(counts, node->right, hypers);
  }
}

double forest_loglik(std::vector<Node*>& forest) {
  double out = 0.0;
  for(int t = 0; t < forest.size(); t++) {
    out += tree_loglik(forest[t]);
  }
  return out;
}

double tree_loglik(Node* node) {
  double out = 0.0;
  if(node->is_leaf) {
    out += log(1.0 - growth_prior(node));
  }
  else {
    out += log(growth_prior(node));
    out += tree_loglik(node->left);
    out += tree_loglik(node->right);
  }
  return out;
}

std::vector<Node*> init_forest(Hypers& hypers) {

  std::vector<Node*> forest(0);
  for(int t = 0; t < hypers.num_trees; t++) {
    Node* n = new Node(&hypers);
    forest.push_back(n);
  }
  return forest;
}
