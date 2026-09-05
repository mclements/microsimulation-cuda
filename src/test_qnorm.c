
#include "nmath.h"
#include <stdio.h>

int main() {
  double x;
  x = qnorm(0.975,0.0,1.0,0,0);
  printf("%f\n",x);
  x = qgamma(0.1,1.0,1.0,0,0);
  printf("%f\n",x);
  x = qpois(0.975,10.0,0,0);
  printf("%f\n",x);
  return 0;
}
