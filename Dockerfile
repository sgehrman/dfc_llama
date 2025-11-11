
FROM ubuntu:22.04

# ===============================================================
# We need to build llama.cpp with an older linux version to ensure compatibility
# with a wide range of systems. Otherwise if I build with a newer glibc, 
# then it will not work on older systems.

# $ docker build -t llama-cpp-build .
# $ docker run -e UID=$(id -u) -e GID=$(id -g) --rm -v "$(pwd)/docker_build:/output" llama-cpp-build

# Interactive shell if you need to debug
# $ docker run --rm -it llama-cpp-build
# ===============================================================

# Install build essentials and CMake
RUN apt update && \
    apt install -y build-essential cmake git wget xz-utils libvulkan-dev

# Install Vulkan SDK
ARG VULKAN_VERSION=1.4.321.1
RUN ARCH=$(uname -m) && \
    wget -qO /tmp/vulkan-sdk.tar.xz https://sdk.lunarg.com/sdk/download/${VULKAN_VERSION}/linux/vulkan-sdk-linux-${ARCH}-${VULKAN_VERSION}.tar.xz && \
    mkdir -p /opt/vulkan && \
    tar -xf /tmp/vulkan-sdk.tar.xz -C /tmp --strip-components=1 && \
    mv /tmp/${ARCH}/* /opt/vulkan/ && \
    rm -rf /tmp/*

# Set environment variables
ENV VULKAN_SDK=/opt/vulkan
ENV PATH=$VULKAN_SDK/bin:$PATH
ENV LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=$VULKAN_SDK:$CMAKE_PREFIX_PATH
ENV PKG_CONFIG_PATH=$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH

# Set working directory
WORKDIR /workspace

# Copy your source code into the container
COPY . /workspace

# Build llama.cpp shared library
WORKDIR /workspace/src/llama.cpp
RUN cmake -B build -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DGGML_VULKAN=1 \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DGGML_BLAS_DEFAULT=ON \
  -DGGML_OPENMP=ON \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE \
  -DCMAKE_INSTALL_RPATH="\$ORIGIN" && \
   cmake --build build --config Release -j$(nproc)

CMD mkdir -p /output && cp /workspace/src/llama.cpp/build/bin/*.so* /output/ && \
  chown -R ${UID:-1000}:${GID:-1000} /output 