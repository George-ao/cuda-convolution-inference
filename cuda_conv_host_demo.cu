#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_conv_interface.h"

int main(int argc, char** argv) {
  int B = 1;
  int M = 4;
  int C = 1;
  int H = 86;
  int W = 86;
  int K = 7;

  // Optional override: B M C H W K
  if (argc == 7) {
    B = std::atoi(argv[1]);
    M = std::atoi(argv[2]);
    C = std::atoi(argv[3]);
    H = std::atoi(argv[4]);
    W = std::atoi(argv[5]);
    K = std::atoi(argv[6]);
  }

  if (H < K || W < K || B <= 0 || M <= 0 || C <= 0 || K <= 0) {
    std::cerr << "Invalid dimensions. Expected positive values with H>=K and W>=K." << std::endl;
    return 1;
  }

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;

  std::vector<float> input(static_cast<size_t>(B) * C * H * W);
  std::vector<float> mask(static_cast<size_t>(M) * C * K * K);
  std::vector<float> output(static_cast<size_t>(B) * M * H_out * W_out, 0.0f);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& v : input) v = dist(rng);
  for (float& v : mask) v = dist(rng);

  GPUInterface gpu;
  float* d_output = nullptr;
  float* d_input = nullptr;
  float* d_mask = nullptr;

  gpu.conv_forward_gpu_prolog(output.data(), input.data(), mask.data(), &d_output, &d_input, &d_mask,
                              B, M, C, H, W, K);
  gpu.conv_forward_gpu(d_output, d_input, d_mask, B, M, C, H, W, K);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  gpu.conv_forward_gpu_epilog(output.data(), d_output, d_input, d_mask, B, M, C, H, W, K);

  const double checksum = std::accumulate(output.begin(), output.end(), 0.0);
  std::cout << "Dims B/M/C/H/W/K: " << B << "/" << M << "/" << C << "/" << H << "/" << W
            << "/" << K << std::endl;
  std::cout << "Output size: " << output.size() << std::endl;
  std::cout << "Output checksum: " << checksum << std::endl;
  std::cout << "First values: ";
  for (size_t i = 0; i < std::min<size_t>(10, output.size()); ++i) {
    std::cout << output[i] << (i + 1 == std::min<size_t>(10, output.size()) ? '\n' : ' ');
  }

  return 0;
}
