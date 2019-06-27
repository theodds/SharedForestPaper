#include "functions.h"

#include <RcppArmadillo.h>

#ifndef M_SQRT_2PI
#define M_SQRT_2PI 2.5066282746310005024157652848
#endif

int sample_class(const arma::vec& probs) {
  double U = R::unif_rand();
  double foo = 0.0;
  int K = probs.size();

  // Sample
  for(int k = 0; k < K; k++) {
    foo += probs(k);
    if(U < foo) {
      return(k);
    }
  }
  return K - 1;
}

int sample_class(int n) {
  double U = R::unif_rand();
  double p = 1.0 / ((double)n);
  double foo = 0.0;

  for(int k = 0; k < n; k++) {
    foo += p;
    if(U < foo) {
      return k;
    }
  }
  return n - 1;
}

int sample_class_col(const arma::sp_mat& probs, int col) {
  double U = R::unif_rand();
  double cumsum = 0.0;

  arma::sp_mat::const_col_iterator it = probs.begin_col(col);
  arma::sp_mat::const_col_iterator it_end = probs.end_col(col);
  for(; it != it_end; ++it) {
    cumsum += (*it);
    if(U < cumsum) {
      return it.row();
    }
  }
  return it.row();

}

double logit(double x) {
  return log(x) - log(1.0-x);
}

double expit(double x) {
  return 1.0 / (1.0 + exp(-x));
}

double log_sum_exp(const arma::vec& x) {
  double M = x.max();
  return M + log(sum(exp(x - M)));
}

// [[Rcpp::export]]
double rlgam(double shape) {
  if(shape >= 0.1) return log(Rf_rgamma(shape, 1.0));

  double a = shape;
  double L = 1.0/a- 1.0;
  double w = exp(-1.0) * a / (1.0 - a);
  double ww = 1.0 / (1.0 + w);
  double z = 0.0;
  do {
    double U = unif_rand();
    if(U <= ww) {
      z = -log(U / ww);
    }
    else {
      z = log(unif_rand()) / L;
    }
    double eta = z >= 0 ? -z : log(w)  + log(L) + L * z;
    double h = -z - exp(-z / a);
    if(h - eta > log(unif_rand())) break;
  } while(true);

  // Rcout << "Sample: " << -z/a << "\n";

  return -z/a;
}

arma::vec rdirichlet(const arma::vec& shape) {
  arma::vec out = arma::zeros<arma::vec>(shape.size());
  for(int i = 0; i < shape.size(); i++) {
    do {
      out(i) = Rf_rgamma(shape(i), 1.0);
    } while(out(i) == 0);
  }
  out = out / arma::sum(out);
  return out;
}

double alpha_to_rho(double alpha, double scale) {
  return alpha / (alpha + scale);
}

double rho_to_alpha(double rho, double scale) {
  return scale * rho / (1.0 - rho);
}

double logpdf_beta(double x, double a, double b) {
  return (a-1.0) * log(x) + (b-1.0) * log(1 - x) - Rf_lbeta(a,b);
}

bool do_mh(double loglik_new, double loglik_old,
           double new_to_old, double old_to_new) {

  double cutoff = loglik_new + new_to_old - loglik_old - old_to_new;

  return log(unif_rand()) < cutoff ? true : false;

}

double randnt(double lower, double upper) {
  if( (lower <= 0 && upper == INFINITY) ||
      (upper >= 0 && lower == -INFINITY) ||
      (lower <= 0 && upper >= 0 && upper - lower > M_SQRT_2PI)) {
    while(true) {
      double r = R::norm_rand();
      if(r > lower && r < upper)
	return(r);
    }
  }
  else if( (lower > 0) &&
	   (upper - lower >
	    2.0 / (lower + sqrt(lower * lower + 4.0)) *
	    exp((lower * lower - lower * sqrt(lower * lower + 4.0)) / 4.0))
	   )
    {
      double a = (lower + sqrt(lower * lower + 4.0))/2.0;
      while(true) {
	double r = R::exp_rand() / a + lower;
	double u = R::unif_rand();
	if (u < exp(-0.5 * pow(r - a, 2)) && r < upper) {
	  return(r);
	}
      }
    }
  else if ( (upper < 0) &&
	    (upper - lower >
	     2.0 / (-upper + sqrt(upper * upper + 4.0)) *
	     exp((upper*upper + upper * sqrt(upper * upper + 4.0)) / 4.0))
	    )
    {
      double a = (-upper + sqrt(upper*upper + 4.0)) / 2.0;
      while(true) {
	double r = R::exp_rand() / a - upper;
	double u = R::unif_rand();
	if (u < exp(-0.5 * pow(r - a, 2)) && r < -lower) {
	  return(-r);
	}
      }
    }
  else {
    while(true) {
      double r = lower + R::unif_rand() * (upper - lower);
      double u = R::unif_rand();
      double rho;
      if (lower > 0) {
	rho = exp((lower*lower - r*r) * 0.5);
      }
      else if (upper < 0) {
	rho = exp((upper*upper - r*r) * 0.5);
      }
      else {
	rho = exp(-r*r * 0.5);
      }
      if(u < rho) {
	return(r);
      }
    }
  }
}

double randnt(double mu, double sigma, double lower, double upper) {
  double Z = randnt( (lower - mu) / sigma,
		     (upper - mu) / sigma
		     );
  return(mu + sigma * Z);
}
