NVCC ?= nvcc
NVCC_ARCH ?= -arch=sm_86
TARGET := gpu_sort

all: $(TARGET)

$(TARGET): main.cpp gpu_sort.cpp gpu_sort.cu gpu_sort.h
	$(NVCC) -std=c++17 $(NVCC_ARCH) -O3 main.cpp gpu_sort.cpp gpu_sort.cu -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.csv

.PHONY: all run clean
