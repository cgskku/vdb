# Compiler settings
# ./kmeans 100000 256 25 10 1536
NVCC = nvcc
CXX = g++

INCLUDES = -I.
CFLAG = -O3

TARGETS = kmeans kmeans_dim kmeans_dim_chunk kmeans_test kmeans-hierarchical

OBJS = kmeans.o kmeans_dim.o kmeans_dim_chunk.o kmeans_test.o kmeans-hierarchical.o

all: $(TARGETS)

kmeans: kmeans.o kmeans.cpp
	$(NVCC) $(INCLUDES) $^ -o $@

kmeans_test: kmeans_test.o kmeans_test.cpp
	$(NVCC) $(INCLUDES) $^ -o $@

kmeans-hierarchical: kmeans-hierarchical.o kmeans-hierarchical.cpp

%.o: %.cu
	$(NVCC) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGETS) *.o

kmeans.o: kmeans.cu kmeans.h
kmeans_dim.o: kmeans_dim.cu kmeans.h
kmeans_dim_chunk.o: kmeans_dim_chunk.cu kmeans.h
kmeans_test.o: kmeans_test.cu kmeans_test.h
kmeans-hierarchical.o: kmeans-hierarchical.cu kmeans-hierarchical.h