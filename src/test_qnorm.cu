
#include "nmath.h"
#include <stdio.h>


int main() {
  printf("%f\n", dnorm(0.0,0.0,1.0,0));
  printf("%f\n", pnorm(1.96,0.0,1.0,1,0));
  printf("%f\n", qnorm(0.975,0.0,1.0,1,0));
  printf("%f\n", qgamma(0.1,1.0,1.0,1,0));
  printf("%f\n", qpois(0.975,10.0,1,0));
  printf("%f\n", unif_rand());
  printf("%f\n", rnorm(0.0,1.0));
  printf("%f\n", rpois(17.0));
  return 0;
}
