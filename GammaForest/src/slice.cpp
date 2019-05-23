#include "slice.h"

double slice_sampler(double x0, Loglik* g, double w, double lower, double upper) {

  /* Find the log density at the initial point, if not already known. */
  double gx0 = g->L(x0);

  /* Determine the slice level, in log terms. */

  double logy = gx0 - exp_rand();

  /* Find the initial interval to sample from */

  double u = w * unif_rand();
  double L = x0 - u;
  double R = x0 + (w-u);

  /* Expand the interval until its ends are outside the slice, or until the
     limit on steps is reached */

  do {

    if(L <= lower) break;
    if(g->L(L) <= logy) break;
    L -= w;

  } while(true);

  do {
    if(R >= upper) break;
    if(g->L(R) <= logy) break;
    R += w;
  } while(true);

  // Shrink interval to lower and upper bounds

  if(L < lower) L = lower;
  if(R > upper) R = upper;

  // Sample from the interval, shrinking it on each rejection

  double x1 = 0.0;

  do {

    x1 = (R - L) * unif_rand() + L;
    double gx1 = g->L(x1);

    if(gx1 >= logy) break;

    if(x1 > x0) {
      R = x1;
    }
    else {
      L = x1;
    }

  } while(true);

  return x1;

}
