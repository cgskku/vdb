NVCC ?= nvcc
NVCC_ARCH ?= -arch=sm_86
TARGET := gpu_sort

all: $(TARGET)

$(TARGET): main.cpp gpu_sort.cpp gpu_sort.cu gpu_sort.h pipeline.cpp pipeline.cu pipeline.h
	$(NVCC) -std=c++17 $(NVCC_ARCH) -O3 -Xcompiler -pthread main.cpp gpu_sort.cpp gpu_sort.cu pipeline.cpp pipeline.cu -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TARGET)
	./run_regression_tests.sh

clean:
	rm -f $(TARGET) *.o *.csv

.PHONY: all run test clean
