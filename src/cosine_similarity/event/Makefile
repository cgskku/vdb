NVCC = nvcc
CXX = g++

INCLUDES = -I.
CFLAGS = -O3

TARGET = cosine
OBJS = cosine.o
LIBS = -lcublas -lcusolver

all: $(TARGET)

$(TARGET): cosine.o cosine.cpp
	$(NVCC) $(INCLUDES) $^ -o $@ $(LIBS)

%.o: %.cu
	$(NVCC) $(INCLUDES) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o

cosine.o: cosine.cu cosine.h
