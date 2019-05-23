#include "recbart.h"

using namespace arma;
using namespace Rcpp;

Node::Node(Hypers* hypers) {
  is_leaf = true;
  is_root = true;
  left    = this;
  right   = this;
  parent  = this;

  var     = 0;
  val     = 0.0;
  lower   = 0.0;
  upper   = 1.0;
  depth   = 0;

  lambda  = 1.0;
  theta   = 0.0;

  this->hypers = hypers;
}

Node::Node(Node* parent) {
  is_leaf = true;
  is_root = false;
  this->parent = parent;
  left = this;
  right = this;

  var = 0;
  val = 0.0;
  lower = 0.0;
  upper = 1.0;
  depth = parent->depth + 1;
  lambda = 1.0;
  theta = 0.0;

  hypers = parent->hypers;
}

void Node::GetLimits() {
  Node* y = this;
  lower = 0.0;
  upper = 1.0;
  bool my_bool = y->is_root ? false : true;
  while(my_bool) {
    bool is_left = y->is_left();
    y = y->parent;
    my_bool = y->is_root ? false : true;
    if(y->var == var) {
      my_bool = false;
      if(is_left) {
        upper = y->val;
        lower = y->lower;
      }
      else {
        upper = y->upper;
        lower = y->val;
      }
    }
  }
}

void Node::BirthLeaves() {
  if(is_leaf) {
    // Rcout << "\nMake left\n";
    left    = new Node(this);
    // Rcout << "Make left\n";
    right   = new Node(this);
    is_leaf = false;
    // Rcout << "Sample var\n";
    var     = hypers->SampleVar();
    // Rcout << "Get limits\n";
    GetLimits();
    // Rcout << "Sample val\n";
    val     = (upper - lower) * unif_rand() + lower;
  }
}

bool Node::is_left() {
  return (this == this->parent->left);
}

void Node::DeleteLeaves() {
  delete left;
  delete right;
  left = this;
  right = this;
  is_leaf = true;
}

Node::~Node() {
  if(!is_leaf) {
    delete left;
    delete right;
  }
}
