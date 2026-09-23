/*
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 2000, 2003  The R Core Team
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/
 *
 */

#include "nmath.h"
#include "dpq.h"

/* A version of Marsaglia-MultiCarry */

struct MarsagliaMultiCarry {
  unsigned int I1, I2;
  HD MarsagliaMultiCarry() { I1=1234, I2=5678; }
};
typedef struct MarsagliaMultiCarry MarsagliaMultiCarry_t;

HD
double unif_rand(MarsagliaMultiCarry_t *rng)
{
    rng->I1= 36969*(rng->I1 & 0177777) + (rng->I1>>16);
    rng->I2= 18000*(rng->I2 & 0177777) + (rng->I2>>16);
    return ((rng->I1 << 16)^(rng->I2 & 0177777)) * 2.328306437080797e-10; /* in [0,1) */
}

#include <math.h>
#include <stdint.h>
//copied from src/main/RNG.c:
//generate a random non-negative integer < 2 ^ bits in 16 bit chunks
static double rbits(int bits, MarsagliaMultiCarry_t *rng)
{
    int_least64_t v = 0;
    for (int n = 0; n <= bits; n += 16) {
	int v1 = (int) floor(unif_rand(rng) * 65536);
	v = 65536 * v + v1;
    }
    // mask out the bits in the result that are not needed
    return (double) (v & ((1L << bits) - 1));
}

double R_unif_index(double dn, MarsagliaMultiCarry_t *rng)
{
    // rejection sampling from integers below the next larger power of two
    if (dn <= 0)
	return 0.0;
    int bits = (int) ceil(log2(dn));
    double dv;
    do { dv = rbits(bits, rng); } while (dn <= dv);
    return dv;
}
