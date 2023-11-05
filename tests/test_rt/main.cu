#include <cstdio>

#include "camera.cuh"
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
__global__ void TestKernel(Camera *camera, TLAS *tlas, float *frame_buffer)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < camera->width() && j < camera->height())
    {
        const uint32_t pixel_index = j * camera->width() + i;
        uint32_t seed = Tea<4>(pixel_index, 0);
        Vec3 color;
        for (uint32_t s = 0; s < camera->spp(); ++s)
        {
            const float u = s * camera->spp_inv(), v = GetVanDerCorputSequence<2>(s + 1),
                        x = 2.0f * (i + u) / camera->width() - 1.0f,
                        y = 1.0f - 2.0f * (j + v) / camera->height();
            const Vec3 look_dir =
                Normalize(camera->front() + x * camera->view_dx() + y * camera->view_dy());
            Hit hit;
            Ray ray = Ray(camera->eye(), look_dir);
            tlas->Intersect(&ray, &hit);
            if (hit.valid)
                // color += hit.normal * 0.5f + Vec3{0.5f};
                color += {hit.texcoord.u, hit.texcoord.v, 0.0f};
            else
                printf("(%ld, %ld), %ld\n", i, j, s);
        }
        color *= camera->spp_inv();
        for (int channel = 0; channel < 3; ++channel)
        {
            color[channel] = fminf(color[channel], 1.0f);
            frame_buffer[pixel_index * 3 + channel] =
                (color[channel] <= 0.0031308f)
                    ? (12.92f * color[channel])
                    : (1.055f * powf(color[channel], 1.0f / 2.4f) - 0.055f);
        }
    }
}
#endif

void TestKernelCpu(Camera *camera, TLAS *tlas, float *frame_buffer)
{
    auto test_pixel = [&](int i, int j)
    {
        const uint32_t pixel_index = j * camera->width() + i;
        uint32_t seed = Tea<4>(pixel_index, 0);
        Vec3 color;
        for (uint32_t s = 0; s < camera->spp(); ++s)
        {
            const float u = s * camera->spp_inv(), v = GetVanDerCorputSequence<2>(s + 1),
                        x = 2.0f * (i + u) / camera->width() - 1.0f,
                        y = 1.0f - 2.0f * (j + v) / camera->height();
            const Vec3 look_dir =
                Normalize(camera->front() + x * camera->view_dx() + y * camera->view_dy());
            Hit hit;
            Ray ray = Ray(camera->eye(), look_dir);
            tlas->Intersect(&ray, &hit);
            if (hit.valid)
                // color += hit.normal * 0.5f + Vec3{0.5f};
                color += {hit.texcoord.u, hit.texcoord.v, 0.0f};
            else
                printf("(%d, %d), %ld\n", i, j, s);
        }
        color *= camera->spp_inv();
        for (int channel = 0; channel < 3; ++channel)
        {
            color[channel] = fminf(color[channel], 1.0f);
            frame_buffer[pixel_index * 3 + channel] =
                (color[channel] <= 0.0031308f)
                    ? (12.92f * color[channel])
                    : (1.055f * powf(color[channel], 1.0f / 2.4f) - 0.055f);
        }
    };

    for (int i = 0; i < camera->width(); ++i)
    {
        for (int j = 0; j < camera->height(); ++j)
        {
            test_pixel(i, j);
        }
    }
}

void Test(BackendType backend_type, const std::string &output_filname)
{
    Scene *scene = new Scene(backend_type);
    Instance::Info info;
    // 0 Floor
    info = {};
    info.type = Instance::Type::kRectangle;
    info.cube.to_world = {{0, 1, 0, 0}, {0, 0, 2, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}};
    scene->AddInstance(info);
    // 1 Ceiling
    info = {};
    info.type = Instance::Type::kRectangle;
    info.cube.to_world = {{-1, 0, 0, 0}, {0, 0, -2, 2}, {0, -1, 0, 0}, {0, 0, 0, 1}};
    scene->AddInstance(info);
    // 2 BackWall
    info = {};
    info.type = Instance::Type::kRectangle;
    info.cube.to_world = {{0, 1, 0, 0}, {1, 0, 0, 1}, {0, 0, -2, -1}, {0, 0, 0, 1}};
    scene->AddInstance(info);
    // 3 RightWall
    info = {};
    info.type = Instance::Type::kRectangle;
    info.cube.to_world = {{0, 0, 2, 1}, {1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 1}};
    scene->AddInstance(info);
    // 4 LeftWall
    info = {};
    info.type = Instance::Type::kRectangle;
    info.cube.to_world = {{0, 0, -2, -1}, {1, 0, 0, 1}, {0, -1, 0, 0}, {0, 0, 0, 1}};
    scene->AddInstance(info);
    // 5 Light
    info = {};
    info.type = Instance::Type::kRectangle;
    info.cube.to_world = {
        {0.235, 0, 0, -0.005}, {0, 0, -0.0893, 1.98}, {0, 0.19, 0, -0.03}, {0, 0, 0, 1}};
    scene->AddInstance(info);
    // 6  Tall Box
    info = {};
    info.type = Instance::Type::kCube;
    info.cube.to_world = {{0.286776, 0.098229, 0, -0.335439},
                          {0, 0, -0.6, 0.6},
                          {-0.0997984, 0.282266, 0, -0.291415},
                          {0, 0, 0, 1}};
    scene->AddInstance(info);
    // 7 Short Box
    info = {};
    info.type = Instance::Type::kCube;
    info.cube.to_world = {{0.0851643, 0.289542, 0, 0.328631},
                          {0, 0, -0.3, 0.3},
                          {-0.284951, 0.0865363, 0, 0.374592},
                          {0, 0, 0, 1}};
    scene->AddInstance(info);

    // 8 Sphere
    info = {};
    info.type = Instance::Type::kSphere;
    info.sphere.center = {0.328631014f, 0.75f, 0.374592006f};
    info.sphere.radius = 0.15f;
    info.sphere.to_world = {};
    scene->AddInstance(info);

    TLAS *tlas = scene->Commit();
    Camera *camera = MallocElement<Camera>(backend_type);
    *camera = Camera();

    float *frame = MallocArray<float>(backend_type, camera->height() * camera->width() * 3);

#ifdef ENABLE_CUDA
    if (backend_type == BackendType::kCpu)
    {
#endif
        TestKernelCpu(camera, tlas, frame);
#ifdef ENABLE_CUDA
    }
    else
    {
        dim3 threads_per_block = {8, 8, 1},
             num_blocks = {static_cast<unsigned int>(camera->width() / 8 + 1),
                           static_cast<unsigned int>(camera->height() / 8 + 1), 1};
        TestKernel<<<num_blocks, threads_per_block>>>(camera, tlas, frame);
        CheckCudaErrors(cudaGetLastError());
        CheckCudaErrors(cudaDeviceSynchronize());
    }
#endif
    image_io::Write(camera->width(), camera->height(), frame, output_filname);

    delete scene;
    DeleteElement(backend_type, camera);
    DeleteArray(backend_type, frame);
}

int main()
{
#ifdef ENABLE_CUDA
    Test(BackendType::kCuda, "result_cuda.png");
    cudaDeviceReset();
#endif
    Test(BackendType::kCpu, "result_cpu.png");
    return 0;
}
