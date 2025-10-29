# K-means vector clustering for Inverted File Index build

This project implements a CUDA-accelerated K-means clustering algorithm for building an inverted file index. 
It leverages `Arrow` and `Parquet` libraries for handling large datasets efficiently.

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

### 2. Build the K-means Program

```bash
nvcc -I<path_to_vcpkg>/installed/x64-linux/include \
    kmeans_openai.cpp kmeans.cu -o kmeans_openai \
    -L<path_to_vcpkg>/installed/x64-linux/lib \
    -larrow -lparquet -lthrift -lcrypto -lzstd -lbrotlidec -lbrotlienc -lbrotlicommon -lbz2 -llz4 -lsnappy -lpthread -lz -ldl
```

### 3. Run the K-means Program

```bash
./kmeans_openai <path_to_data_file> <threads_per_block> <norm_type>
```
Example
```bash
./kmeans_openai /path/to/data/shuffle_train-00-of-10.parquet 256 cosine
``