# Compiler settings
# ./kmeans 100000 256 25 10 1536
NVCC = nvcc
CXX = g++

INCLUDES = -I.
CFLAG = -O3

TARGETS = kmeans  kmeans_dim kmeans_dim_chunk

OBJS = kmeans.o kmeans_dim.o kmeans_dim_chunk.o

all: $(TARGETS)

kmeans: kmeans.o kmeans.cpp
	$(NVCC) $(INCLUDES) $^ -o $@

kmeans_dim: kmeans_dim.o kmeans.cpp
	$(NVCC) $(INCLUDES) $^ -o $@

kmeans_dim_chunk: kmeans_dim_chunk.o kmeans_chunk.cpp
	$(NVCC) $(INCLUDES) $^ -o $@

%.o: %.cu
	$(NVCC) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o

kmeans.o: kmeans.cu kmeans.h
kmeans_dim.o: kmeans_dim.cu kmeans.h
kmeans_dim_chunk.o: kmeans_dim_chunk.cu kmeans.h