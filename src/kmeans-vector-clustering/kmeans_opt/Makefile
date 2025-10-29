# Compiler settings
NVCC = nvcc
CXX = g++

# Include paths
INCLUDES = -std=c++17 -I. -I.. -I../../../vcpkg/installed/x64-linux/include

# Library paths and libraries
LIB_PATHS = -L../../../vcpkg/installed/x64-linux/lib
LIBS = -larrow -lparquet -lthrift -lcrypto -lzstd -lbrotlidec -lbrotlienc -lbrotlicommon -lbz2 -llz4 -lsnappy -lpthread -lz -ldl

# Compiler flags
CFLAGS = -O3

# Targets
#TARGETS = kmeans_chunk_parquet kmeans_bank_parquet kmeans_parquet
TARGETS = kmeans_parquet

# Object files
OBJS = 

all: $(TARGETS)

# # kmeans_chunk_parquet
# kmeans_chunk_parquet: kmeans_chunk_parquet.cpp 
# 	$(NVCC) $(INCLUDES) $(CFLAGS) kmeans_chunk_parquet.cpp cuda/kmeans_dim_chunk_no_tile.cu -o $@ $(LIB_PATHS) $(LIBS)

# # kmeans_bank_parquet
# kmeans_bank_parquet: kmeans_bank_parquet.cpp
# 	$(NVCC) $(INCLUDES) $(CFLAGS) kmeans_bank_parquet.cpp cuda/kmeans_dim_bank.cu -o $@ $(LIB_PATHS) $(LIBS)

# kmeans_parquet
kmeans_parquet: kmeans_parquet.cpp
	$(NVCC) $(INCLUDES) $(CFLAGS) kmeans_parquet.cpp cuda/kmeans.cu -o $@ $(LIB_PATHS) $(LIBS)


# Clean targets
clean:
	rm -f $(TARGETS) *.o

# Clean all including executables
clean-all:
	rm -f $(TARGETS) *.o kmeans_chunk_parquet kmeans_bank_parquet kmeans_parquet

# Help target
help:
	@echo "Available targets:"
	@echo "  all                    - Build all targets"
	@echo "  kmeans_chunk_parquet   - Build kmeans_chunk_parquet"
	@echo "  kmeans_bank_parquet    - Build kmeans_bank_parquet"
	@echo "  kmeans_mini_parquet    - Build kmeans_mini_parquet"
	@echo "  kmeans_mmap_parquet    - Build kmeans_mmap_parquet"
	@echo "  kmeans_parquet         - Build kmeans_parquet"
	@echo "  clean                  - Remove object files"
	@echo "  clean-all              - Remove all generated files"
	@echo "  help                   - Show this help message"

.PHONY: all clean clean-all help
