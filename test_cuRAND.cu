/**
 * @brief Implementation of RngStreams using curandStateMRG32k3a_t
 *
 * Mark Clements
 * 2026-09-20
 * Licence: GPL>=3 (set_seed() is from R)
 */

#include <cstdio>
#include <curand_kernel.h>

// Assumes the following internal struct for curandStateMRG32k3a_t:
//
// struct curandStateMRG32k3a {
//     unsigned int s1[3];
//     unsigned int s2[3];
//     int boxmuller_flag;
//     int boxmuller_flag_double;
//     float boxmuller_extra;
//     double boxmuller_extra_double;
// };
//
// typedef struct curandStateMRG32k3a curandStateMRG32k3a_t;

struct RngStream : curandStateMRG32k3a_t {
  unsigned int start_sequence[6];
  unsigned int start_subsequence[6];

  __host__ __device__
  RngStream();
  
  __host__ __device__
  RngStream(unsigned int seed[6]);
  
};

__host__ __device__
RngStream::RngStream() : curandStateMRG32k3a_t()
{
  for (int i=0; i<3; i++) 
    s1[i] = s2[i] = 12345;
  for (int i=0; i<6; i++) 
    start_sequence[i] = start_subsequence[i] = 12345;
}

__host__ __device__
RngStream::RngStream(unsigned int seed[6]) : curandStateMRG32k3a_t()
{
  for (int i=0; i<3; i++) {
    s1[i] = seed[i];
    s2[i] = seed[i+3];
  }
  for (int i=0; i<6; i++) 
    start_sequence[i] = start_subsequence[i] = seed[i];
}

__host__ __device__
void set_seed(uint32_t seed, RngStream *rng) 
{
  for (int j = 0; j < 50; ++j)
    seed = (69069 * seed + 1);
  for (int i = 0; i < 6; ++i) {
    seed = 69069 * seed + 1;
    while (seed >= MRG32K3A_MOD2) seed = 69069 * seed + 1;
    if (i<3)
      rng->s1[i] = seed;
    else
      rng->s2[i-3] = seed;
    rng->start_sequence[i] = rng->start_subsequence[i] = seed;
  }
}

__host__ __device__
void reset_sequence(RngStream *rng) {
  for (int i=0; i<3; i++) {
    rng->s1[i] = rng->start_sequence[i];
    rng->s2[i] = rng->start_sequence[i+3];
  }
  for (int i=0; i<6; i++)
    rng->start_subsequence[i] = rng->start_sequence[i];
  rng->boxmuller_flag = rng->boxmuller_flag_double = 0;
  rng->boxmuller_extra = 0;
  rng->boxmuller_extra_double = 0;
}

__host__ __device__
void reset_subsequence(RngStream *rng) {
  for (int i=0; i<3; i++) {
    rng->s1[i] = rng->start_subsequence[i];
    rng->s2[i] = rng->start_subsequence[i+3];
  }
  rng->boxmuller_flag = rng->boxmuller_flag_double = 0;
  rng->boxmuller_extra = 0;
  rng->boxmuller_extra_double = 0;
}

__device__
void reset_skipahead_subsequence(RngStream *rng, unsigned long long n = 1) {
  reset_subsequence(rng);
  skipahead_subsequence(n,rng); // device-only?!
  for (int i=0; i<3; i++) {
    rng->start_subsequence[i] = rng->s1[i];
    rng->start_subsequence[i+3] = rng->s2[i];
  }
};

__device__
void reset_skipahead_sequence(RngStream *rng, unsigned long long n = 1) {
  reset_sequence(rng);
  skipahead_sequence(n,rng); // device-only?!
  for (int i=0; i<3; i++) {
    rng->start_sequence[i] = rng->start_subsequence[i] = rng->s1[i];
    rng->start_sequence[i+3] = rng->start_subsequence[i+3] = rng->s2[i];
  }
}

__global__ void test()
{
  double u;
  RngStream rng;

  u = curand_uniform_double(&rng);
  printf("%.17g\n", u);
    
  reset_skipahead_subsequence(&rng);
  u = curand_uniform_double(&rng);
  printf("%.17g\n", u);
      
  reset_skipahead_sequence(&rng);
  u = curand_uniform_double(&rng);
  printf("%.17g\n", u);

  set_seed(12345, &rng);
  u = curand_uniform_double(&rng);
  printf("%.17g\n", u);
    
}

int main()
{
  
  test<<<1,1>>>();
  cudaDeviceSynchronize();
    
}
