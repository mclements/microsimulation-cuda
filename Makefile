NVCC    := nvcc
NVFLAGS := -arch=sm_61 -ccbin g++-11 -DMATHLIB_STANDALONE=1 -Iinclude -Isrc
CC      := gcc
CFLAGS  := -DMATHLIB_STANDALONE=1 -Iinclude -Isrc
LDLIBS  := -lm

TARGET := test_qnorm

SRC := $(wildcard src/*.cu)
SRC := $(filter-out src/test_qnorm.cu,$(SRC))

OBJ := $(SRC:.cu=.o)

$(TARGET): src/test_qnorm.cu $(OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDLIBS)

test_RngStream: test_RngStream.cu
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDLIBS)

test_RngStream-2: test_RngStream-2.cu
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDLIBS)

test_cuRAND: test_cuRAND.cu
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJ) test_RngStream

.PHONY: clean

%.o: %.cu
	$(NVCC) -c $(NVFLAGS) $< -o $@
