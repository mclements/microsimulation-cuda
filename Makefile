NVCC     = nvcc
NVFLAGS  = -arch=sm_61 -ccbin g++-11
CPP      = g++
CPPFLAGS =

test_RngStream: test_RngStream.cu

.PHONY: clean run

clean:
	rm -f test_RngStream

%: %.cu
	$(NVCC) $(NVFLAGS) $< -o $@

%: %.cpp
	$(CPP) $(CPPFLAGS) $< -o $@

