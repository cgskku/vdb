# ./program <num_points> <k-cluster> <iteration> <dimension>
# Example usage: ./kmeans_dim_pca 1000000 100 10 1536

NVCC = nvcc
CXX = g++

INCLUDES = -I.
CFLAGS = -O3

TARGET = kmeans_dim_pca test kmeans_test
OBJS = kmeans_dim_pca.o test.o kmeans_test.o
LIBS = -lcublas -lcusolver

all: $(TARGET)

kmeans_dim_pca: kmeans_dim_pca.o kmeans_pca.o
	$(NVCC) $(INCLUDES) $^ -o $@ $(LIBS)

test: test.o tmp.o
	$(NVCC) $(INCLUDES) $^ -o $@ $(LIBS)

kmeans_test: kmeans_test.o test.o
	$(NVCC) $(INCLUDES) $^ -o $@ $(LIBS)

%.o: %.cu
	$(NVCC) $(INCLUDES) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(INCLUDES) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o *.txt

kmeans_dim_pca.o: kmeans_dim_pca.cu kmeans.h
kmeans_pca.o: kmeans_pca.cpp kmeans.h
test.o: test.cu kmeans.h
tmp.o: tmp.cpp kmeans.h
kmeans_test.o: kmeans_test.cpp kmeans.h
