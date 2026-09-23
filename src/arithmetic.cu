
#include "nmath.h"

/* should return x for very small x (e.g. expm1(x))*/
HD
double f_x_x(double x, double(*f)(double), double m)
{
    return (fabs(x) <= m) ? x : f(x);
}

HD
double Rexpm1(double x)
{
    /* expm1(x) = exp(x) - 1 =  x + x^2/2 + O(x^3)
     *                       =. x  when  x^2/2 < |x| * D_EPS/2
     *                             <==>  |x|   <  D_EPS */
    return f_x_x(x, expm1, DBL_EPSILON);
}

HD
double Rlog1p(double x)
{
    /* log1p(x) = log(1 + x) =  x - x^2/2  + O(x^3)
     *                       =. x  when  |x| < D_EPS, see Rexpm1() */
    return f_x_x(x, log1p, DBL_EPSILON);
}
