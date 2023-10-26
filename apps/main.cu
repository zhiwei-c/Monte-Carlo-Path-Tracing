#include "ray_tracer.cuh"

#ifdef ENABLE_CUDA
#define CheckCudaErrors(val) CheckCuda((val), #val, __FILE__, __LINE__)

inline void CheckCuda(cudaError_t result, char const *const func, const char *const file,
                      const int line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error = %llu at %s:%d inside function '%s' \n",
                static_cast<uint64_t>(result), file, line, func);
        cudaDeviceReset();
        exit(0);
    }
}
#endif

#ifdef ENABLE_CUDA
__global__ void TestKernel(uint32_t num, rt::Vec2 *vec)
{

    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num)
    {
        vec[id] = {static_cast<float>(id)};
        printf("%f\n", vec[id].y);
    }
}
#endif

int main()
{
    rt::Vec2 tmp = {1, 2};

#ifdef ENABLE_CUDA
    uint32_t num = 10;
    rt::Vec2 *vb = nullptr;
    CheckCudaErrors(cudaMallocManaged(&vb, num * sizeof(vb)));
    dim3 threads_per_block_ = {32, 1, 1},
         num_blocks_ = {num / 32 + 1, 1, 1};
    TestKernel<<<num_blocks_, threads_per_block_>>>(num, vb);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaFree(vb));
#endif
    return 0;
}
