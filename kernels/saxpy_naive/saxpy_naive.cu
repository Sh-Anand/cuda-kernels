#include <cuda_runtime.h>
#include <stdio.h>

// Naive SAXPY CUDA kernel using vector types
__global__ void saxpy_naive(float4 *x, float4 *y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i].x = a * x[i].x + y[i].x;
        y[i].y = a * x[i].y + y[i].y;
        y[i].z = a * x[i].z + y[i].z;
        y[i].w = a * x[i].w + y[i].w;
    }
}

int main(int argc, char** argv){
  int n = 1024;
  float a = 2.0f;
  float4* x = new float4[n];
  float4* y = new float4[n];
  for (int i = 0; i < n; i++){
    x[i] = make_float4(i, i+1, i+2, i+3);
  }
  float4* d_x;
  float4* d_y;
  cudaMalloc(&d_x, n * sizeof(float4));
  cudaMalloc(&d_y, n * sizeof(float4));
  cudaMemcpy(d_x, x, n * sizeof(float4), cudaMemcpyHostToDevice);
  saxpy_naive<<<1, 1024>>>(d_x, d_y, a, n);
  cudaMemcpy(y, d_y, n * sizeof(float4), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++){
    printf("y[%d] = (%f, %f, %f, %f)\n", i, y[i].x, y[i].y, y[i].z, y[i].w);
  }
  cudaFree(d_x);
  cudaFree(d_y);
  delete[] x;
  return 0;
}