# Load-manager for Efficient Vector Processing

This project implements a **GPU-accelerated Load Manager** designed to efficiently handle large-scale vector data for distance computation and clustering tasks.  
It introduces a **tile-based memory management system** that dynamically loads and processes vector subsets, preventing GPU memory overflow and maximizing throughput.

## Quickstart

Follow these steps to set up and run the project:

### 1. Clone the Repository and Install Dependencies

```bash
# Clone the vcpkg package manager repository
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
./bootstrap-vcpkg.sh

# Install dependencies for vcpkg
sudo apt-get install flex
sudo apt-get install bison

# Install necessary libraries via vcpkg
./vcpkg install arrow parquet
```

### 2. Build the Load-manager Program

```bash
nvcc -I<path_to_vcpkg>/installed/x64-linux/include \
    load-manager.cpp load-manager.cu -o load-manager \
    -L<path_to_vcpkg>/installed/x64-linux/lib \
    -larrow -lparquet -lthrift -lcrypto -lzstd -lbrotlidec -lbrotlienc -lbrotlicommon -lbz2 -llz4 -lsnappy -lpthread -lz -ldl -lcublas
```

### 3. Run the Load-manager Program

```bash
./load-manager <path_to_data_file> <threads_per_block> <norm_type>
```
Example
```bash
./load-manager /path/to/data/shuffle_train-00-of-10.parquet 256 cosine
```