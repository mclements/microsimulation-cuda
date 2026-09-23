#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "RngStream.cuh"

// using namespace rngstream;

__global__
void test_kernel(
		 rngstream::RngStream* states,
		 float* output,
		 int n_per_thread)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  auto st = states[tid];

  for (int j = 0; j < n_per_thread; ++j) {
    output[tid * n_per_thread + j] = rngstream::U01f(st); // simulate:)
    rngstream::ResetNextSubstream(st);
  }

  states[tid] = st;
}

int main()
{
  constexpr int NTHREAD = 8;
  constexpr int NPER    = 2;

  rngstream::Seed seed =
    {
      12345U, 12345U, 12345U,
      12345U, 12345U, 12345U
    };

  //----------------------------------------------------------------
  // host state
  //----------------------------------------------------------------

  std::vector<rngstream::RngStream> h_states;

  rngstream::RngStream st(seed);
    
  for (int i = 0; i < NTHREAD; ++i) {
      
    h_states.emplace_back(st);
    rngstream::AdvanceSubstreams(st, 1000);

  }

  //----------------------------------------------------------------
  // device memory
  //----------------------------------------------------------------

  rngstream::RngStream* d_states = nullptr;
  float*        d_output = nullptr;

  cudaMalloc(&d_states,
	     sizeof(rngstream::RngStream) * NTHREAD);

  cudaMalloc(&d_output,
	     sizeof(float) * NTHREAD * NPER);

  cudaMemcpy(
	     d_states,
	     h_states.data(),
	     sizeof(rngstream::RngStream) * NTHREAD,
	     cudaMemcpyHostToDevice);
  
  //----------------------------------------------------------------
  // run
  //----------------------------------------------------------------

  test_kernel<<<1,NTHREAD>>>(
			     d_states,
			     d_output,
			     NPER);

  cudaDeviceSynchronize();

  //----------------------------------------------------------------
  // copy back
  //----------------------------------------------------------------

  std::vector<float> h_output(
			       NTHREAD * NPER);

  cudaMemcpy(
	     h_output.data(),
	     d_output,
	     sizeof(float) * NTHREAD * NPER,
	     cudaMemcpyDeviceToHost);

  //----------------------------------------------------------------
  // print
  //----------------------------------------------------------------

  for (int tid = 0;
       tid < NTHREAD;
       ++tid)
    {
      printf("substream %d:\n", tid);

      for (int j = 0;
	   j < NPER;
	   ++j)
        {
	  printf("  %.6f\n",
		 h_output[tid * NPER + j]);
        }

      printf("\n");
    }

  //----------------------------------------------------------------
  // cleanup
  //----------------------------------------------------------------

  cudaFree(d_states);
  cudaFree(d_output);

  // check host code
  printf("Default constructor:\n");
  st = rngstream::RngStream();
  printf("%f\n", rngstream::U01(st));
  printf("As per R's set.seed(12345):\n");
  st = rngstream::RngStream(12345U);
  printf("%f\n", rngstream::U01(st));
  
  return 0;
}
