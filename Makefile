NVCC     = nvcc
NVFLAGS  = -arch=sm_61 -ccbin g++-11
CPP      = g++
CPPFLAGS =

test_RngStream: test_RngStream.cu

.PHONY: clean run

clean:
	rm -f test_RngStream

src/qnorm: src/qnorm.c
	cd src && \
	gcc -c d1mach.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c i1mach.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c arithmetic.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c toms708.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c fmax2.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c log1p.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c cospi.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c gamma.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c lgamma.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c lgammacor.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c chebyshev.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c bd0.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c stirlerr.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c dpois.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c ppois.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c qpois.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c qnorm.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c pnorm.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c dnorm.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c dgamma.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c qgamma.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c pgamma.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc -c mlutils.c -DMATHLIB_STANDALONE=1 -I../include -I. && \
	gcc test_qnorm.c -DMATHLIB_STANDALONE=1 -I../include -I. i1mach.o d1mach.o mlutils.o toms708.o lgammacor.o chebyshev.o stirlerr.o bd0.o log1p.o cospi.o dpois.o ppois.o qpois.o fmax2.o gamma.o lgamma.o pgamma.o qgamma.o dgamma.o pnorm.o dnorm.o qnorm.o -lm

%: %.cu
	$(NVCC) $(NVFLAGS) $< -o $@

%: %.cpp
	$(CPP) $(CPPFLAGS) $< -o $@

