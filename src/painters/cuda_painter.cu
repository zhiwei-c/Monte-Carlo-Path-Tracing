#include "painters.cuh"

#ifdef ENABLE_CUDA

#include <algorithm>
#include <cstdio>
#include <vector>

#include "../utils/math.cuh"
#include "../utils/misc.cuh"
#include "../utils/image_io.cuh"
#include "../utils/timer.cuh"

namespace
{
    __global__ void CreateTextureBuffer(size_t num_texture, Texture::Info *texture_info_buffer,
                                        Texture **texture_buffer)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_texture)
        {
            switch (texture_info_buffer[id].type)
            {
            case Texture::Type::kCheckerboard:
                texture_buffer[id] = new CheckerboardTexture(
                    id, texture_info_buffer[id].data.checkerboard);
                break;
            case Texture::Type::kConstant:
                texture_buffer[id] = new ConstantTexture(
                    id, texture_info_buffer[id].data.constant);
                break;
            case Texture::Type::kBitmap:
                texture_buffer[id] = new Bitmap(id, texture_info_buffer[id].data.bitmap);
                break;
            }
        }
    }

    __global__ void DeleteTextureBuffer(size_t num_texture, Texture **texture_buffer)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_texture)
        {
            SAFE_DELETE_ELEMENT(texture_buffer[id]);
        }
    }

    __global__ void CreateBsdfBuffer(size_t num_bsdf, Bsdf::Info *bsdf_info_buffer,
                                     float *brdf_avg_buffer, float *albedo_avg_buffer,
                                     Bsdf **bsdf_buffer)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_bsdf)
        {
            switch (bsdf_info_buffer[id].type)
            {
            case Bsdf::Type::kAreaLight:
                bsdf_buffer[id] = new AreaLight(id, bsdf_info_buffer[id].data);
                break;
            case Bsdf::Type::kDiffuse:
                bsdf_buffer[id] = new Diffuse(id, bsdf_info_buffer[id].data);
                break;
            case Bsdf::Type::kRoughDiffuse:
                bsdf_buffer[id] = new RoughDiffuse(id, bsdf_info_buffer[id].data);
                break;
            case Bsdf::Type::kConductor:
                bsdf_buffer[id] = new Conductor(id, bsdf_info_buffer[id].data);
                break;
            case Bsdf::Type::kDielectric:
                bsdf_buffer[id] = new Dielectric(id, bsdf_info_buffer[id].data);
                break;
            case Bsdf::Type::kThinDielectric:
                bsdf_buffer[id] = new ThinDielectric(id, bsdf_info_buffer[id].data);
                break;
            case Bsdf::Type::kPlastic:
                bsdf_buffer[id] = new Plastic(id, bsdf_info_buffer[id].data);
                break;
            }
            bsdf_buffer[id]->SetKullaConty(brdf_avg_buffer, albedo_avg_buffer);
        }
    }

    __global__ void DeleteBsdfBuffer(size_t num_bsdf, Bsdf **bsdf_buffer)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_bsdf)
        {
            SAFE_DELETE_ELEMENT(bsdf_buffer[id]);
        }
    }

    __global__ void CreateInstanceBuffer(size_t num_instance,
                                         Instance::Info *instance_info_buffer,
                                         Primitive *primirive_buffer,
                                         Instance *instance_buffer)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_instance)
            instance_buffer[id] = Instance(id, instance_info_buffer[id], primirive_buffer);
    }

    __global__ void CreateEmitterBuffer(size_t num_emitter, Emitter::Info *emitter_info_buffer,
                                        Sun *sun, Emitter **emitter_buffer)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_emitter)
        {
            switch (emitter_info_buffer[id].type)
            {
            case Emitter::Type::kSpot:
                emitter_buffer[id] = new SpotLight(id, emitter_info_buffer[id].data.spot);
                break;
            case Emitter::Type::kDirectional:
                emitter_buffer[id] = new DirectionalLight(id,
                                                          emitter_info_buffer[id].data.directional);
                break;
            case Emitter::Type::kSun:
                emitter_buffer[id] = new Sun(id, emitter_info_buffer[id].data.sun);
                sun = (Sun *)emitter_buffer[id];
                break;
            }
        }
    }

    __global__ void DeleteEmitterBuffer(size_t num_emitter, Emitter **emitter_buffer)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < num_emitter)
        {
            SAFE_DELETE_ELEMENT(emitter_buffer[id]);
        }
    }

    __global__ void CreateAccel(Primitive *primitive_buffer, BvhNode *bvh_node_buffer,
                                Accel *accel)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id == 0)
        {
            *accel = Accel(primitive_buffer, bvh_node_buffer);
        }
    }

    __global__ void CreateIntegrator(float *pixel_buffer, Texture **texture_buffer,
                                     Bsdf **bsdf_buffer, Primitive *primitive_buffer,
                                     Instance *instance_buffer, Accel *accel,
                                     uint32_t num_emitter, Emitter **emitter_buffer,
                                     uint32_t num_area_light, uint32_t *area_light_id_buffer,
                                     EnvMap *env_map, Sun *sun, Integrator *integrator)
    {
        size_t id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id == 0)
        {
            *integrator = Integrator(pixel_buffer, texture_buffer, bsdf_buffer, primitive_buffer,
                                     instance_buffer, accel, num_emitter, emitter_buffer,
                                     num_area_light, area_light_id_buffer, env_map, sun);
        }
    }

    __global__ void DispatchRays(Camera *camera, Integrator *integrator, float *frame_buffer)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < camera->width() && j < camera->height())
        {
            const uint32_t pixel_index = j * camera->width() + i;
            uint32_t seed = Tea(pixel_index, 0, 4);
            Vec3 color;
            for (uint32_t s = 0; s < camera->spp(); ++s)
            {
                const float u = s * camera->spp_inv(),
                            v = GetVanDerCorputSequence(s + 1, 2),
                            x = 2.0f * (i + u) / camera->width() - 1.0f,
                            y = 1.0f - 2.0f * (j + v) / camera->height();
                const Vec3 look_dir = Normalize(camera->front() + x * camera->view_dx() +
                                                y * camera->view_dy());
                color += integrator->GenerateRay(camera->eye(), look_dir, &seed);
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

} // namespace

CudaPainter::CudaPainter(BvhBuilder::Type bvh_type, const SceneInfo &info)
{
    fprintf(stderr, "[info] setup scene ...\n");

    // 创建相机
    camera_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&camera_, sizeof(Camera)));
    CheckCudaErrors(cudaMemcpy(camera_, &info.camera, sizeof(Camera), cudaMemcpyHostToDevice));

    // 创建纹理
    texture_buffer_ = nullptr;
    num_texture_ = info.texture_info_buffer.size();
    CheckCudaErrors(cudaMallocManaged(&texture_buffer_, num_texture_ * sizeof(Texture *)));
    Texture::Info *texture_info_buffer = nullptr;
    CheckCudaErrors(cudaMallocManaged(&texture_info_buffer, num_texture_ * sizeof(Texture::Info)));
    CheckCudaErrors(cudaMemcpy(texture_info_buffer, info.texture_info_buffer.data(),
                               num_texture_ * sizeof(Texture::Info), cudaMemcpyHostToDevice));
    threads_per_block_ = {32, 1, 1};
    num_blocks_ = {static_cast<unsigned int>(num_texture_ / 32 + 1), 1, 1};
    ::CreateTextureBuffer<<<num_blocks_, threads_per_block_>>>(num_texture_,
                                                               texture_info_buffer,
                                                               texture_buffer_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(texture_info_buffer));

    // 创建 Kulla-Conty LUT
    std::vector<float> brdf_avg_buffer(Bsdf::kLutResolution * Bsdf::kLutResolution),
        albedo_avg_buffer(Bsdf::kLutResolution);
    Bsdf::ComputeKullaConty(brdf_avg_buffer.data(), albedo_avg_buffer.data());

    brdf_avg_buffer_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&brdf_avg_buffer_, brdf_avg_buffer.size() * sizeof(float)));
    CheckCudaErrors(cudaMemcpy(brdf_avg_buffer_, brdf_avg_buffer.data(),
                               brdf_avg_buffer.size() * sizeof(float), cudaMemcpyHostToDevice));

    albedo_avg_buffer_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&albedo_avg_buffer_, albedo_avg_buffer.size() * sizeof(float)));
    CheckCudaErrors(cudaMemcpy(albedo_avg_buffer_, albedo_avg_buffer.data(),
                               albedo_avg_buffer.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 创建 BSDF
    num_bsdf_ = info.bsdf_info_buffer.size();

    bsdf_buffer_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&bsdf_buffer_, num_bsdf_ * sizeof(Bsdf *)));

    Bsdf::Info *bsdf_info_buffer = nullptr;
    CheckCudaErrors(cudaMallocManaged(&bsdf_info_buffer, num_bsdf_ * sizeof(Bsdf::Info)));
    CheckCudaErrors(cudaMemcpy(bsdf_info_buffer, info.bsdf_info_buffer.data(),
                               num_bsdf_ * sizeof(Bsdf::Info), cudaMemcpyHostToDevice));

    threads_per_block_ = {32, 1, 1};
    num_blocks_ = {static_cast<unsigned int>(num_bsdf_ / 32 + 1), 1, 1};
    ::CreateBsdfBuffer<<<num_blocks_, threads_per_block_>>>(num_bsdf_, bsdf_info_buffer,
                                                            brdf_avg_buffer_,
                                                            albedo_avg_buffer_,
                                                            bsdf_buffer_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(bsdf_info_buffer));

    //
    // 创建图元
    //
    num_primitive_ = info.primitive_info_buffer.size();
    fprintf(stderr, "[info] total primitive num : %lu\n", num_primitive_);

    std::vector<Primitive> primitive_buffer(num_primitive_);
    std::vector<AABB> aabb_buffer(num_primitive_);
    for (uint32_t i = 0; i < num_primitive_; ++i)
    {
        primitive_buffer[i] = Primitive(i, info.primitive_info_buffer[i]);
        aabb_buffer[i] = primitive_buffer[i].aabb();
    }
    primitive_buffer_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&primitive_buffer_, num_primitive_ * sizeof(Primitive)));
    CheckCudaErrors(cudaMemcpy(primitive_buffer_, primitive_buffer.data(),
                               num_primitive_ * sizeof(Primitive), cudaMemcpyHostToDevice));


    // 创建用于加速计算的数据结构
    bvh_node_buffer_ = nullptr;
    accel_ = nullptr;
    if (num_primitive_ > 0)
    {
        std::vector<BvhNode> bvh_node_buffer;
        switch (bvh_type)
        {
        case BvhBuilder::Type::kNormal:
        {
            NormalBvhBuilder builder;
            builder.Build(num_primitive_, aabb_buffer.data(), &bvh_node_buffer);
            break;
        }
        case BvhBuilder::Type::kLinear:
        {
            LinearBvhBuilder builder;
            builder.Build(num_primitive_, aabb_buffer.data(), &bvh_node_buffer);
            break;
        }
        }

        uint32_t num_bvh_node = bvh_node_buffer.size();
        CheckCudaErrors(cudaMallocManaged(&bvh_node_buffer_, num_bvh_node * sizeof(BvhNode)));
        CheckCudaErrors(cudaMemcpy(bvh_node_buffer_, bvh_node_buffer.data(),
                                   num_bvh_node * sizeof(BvhNode), cudaMemcpyHostToDevice));

        CheckCudaErrors(cudaMallocManaged(&accel_, sizeof(Accel)));
        threads_per_block_ = {1, 1, 1};
        num_blocks_ = {1, 1, 1};
        ::CreateAccel<<<num_blocks_, threads_per_block_>>>(primitive_buffer_, bvh_node_buffer_,
                                                           accel_);
        CheckCudaErrors(cudaGetLastError());
        CheckCudaErrors(cudaDeviceSynchronize());
    }

    // 创建物体实例
    uint32_t num_instance = info.instance_info_buffer.size();
    instance_buffer_ = nullptr;
    CheckCudaErrors(cudaMallocManaged(&instance_buffer_, num_instance * sizeof(Instance)));

    Instance::Info *instance_info_buffer = nullptr;
    CheckCudaErrors(cudaMallocManaged(&instance_info_buffer,
                                      num_instance * sizeof(Instance::Info)));
    CheckCudaErrors(cudaMemcpy(instance_info_buffer, info.instance_info_buffer.data(),
                               num_instance * sizeof(Instance::Info), cudaMemcpyHostToDevice));

    threads_per_block_ = {32, 1, 1};
    num_blocks_ = {static_cast<unsigned int>(num_instance / 32 + 1), 1, 1};
    ::CreateInstanceBuffer<<<num_blocks_, threads_per_block_>>>(num_instance,
                                                                instance_info_buffer,
                                                                primitive_buffer_,
                                                                instance_buffer_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(instance_info_buffer));

    // 处理面光源信息
    std::vector<uint32_t> area_light_id_buffer;
    for (uint32_t i = 0; i < num_instance; ++i)
    {
        if (info.instance_info_buffer[i].is_emitter)
            area_light_id_buffer.push_back(i);
    }
    num_area_light_ = area_light_id_buffer.size();
    area_light_id_buffer_ = nullptr;
    if (!area_light_id_buffer.empty())
    {
        CheckCudaErrors(cudaMallocManaged(&area_light_id_buffer_, num_area_light_ * sizeof(uint32_t)));
        CheckCudaErrors(cudaMemcpy(area_light_id_buffer_, area_light_id_buffer.data(),
                                   num_area_light_ * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    //
    // 创建除面光源之外的其它光源
    //
    env_map_ = nullptr;
    if (info.env_map)
    {
        CheckCudaErrors(cudaMallocManaged(&env_map_, sizeof(EnvMap)));
        CheckCudaErrors(cudaMemcpy(env_map_, info.env_map, sizeof(EnvMap), cudaMemcpyHostToDevice));
    }

    sun_ = nullptr;
    emitter_buffer_ = nullptr;
    num_emitter_ = info.emitter_info_buffer.size();
    CheckCudaErrors(cudaMallocManaged(&emitter_buffer_, num_emitter_ * sizeof(Emitter *)));
    Emitter::Info *emitter_info_buffer = nullptr;
    CheckCudaErrors(cudaMallocManaged(&emitter_info_buffer, num_emitter_ * sizeof(Emitter::Info)));
    CheckCudaErrors(cudaMemcpy(emitter_info_buffer, info.emitter_info_buffer.data(),
                               num_emitter_ * sizeof(Emitter::Info), cudaMemcpyHostToDevice));
    threads_per_block_ = {32, 1, 1};
    num_blocks_ = {static_cast<unsigned int>(num_emitter_ / 32 + 1), 1, 1};
    ::CreateEmitterBuffer<<<num_blocks_, threads_per_block_>>>(num_emitter_, emitter_info_buffer,
                                                               sun_, emitter_buffer_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(emitter_info_buffer));

    // 创建位图纹理引用的像素
    pixel_buffer_ = nullptr;
    if (!info.pixel_buffer.empty())
    {
        uint32_t num_pixel = info.pixel_buffer.size();
        CheckCudaErrors(cudaMallocManaged(&pixel_buffer_, num_pixel * sizeof(float)));
        CheckCudaErrors(cudaMemcpy(pixel_buffer_, info.pixel_buffer.data(),
                                   num_pixel * sizeof(float), cudaMemcpyHostToDevice));
    }

    // 创建积分器
    CheckCudaErrors(cudaMallocManaged(&integrator_, sizeof(Integrator)));
    threads_per_block_ = {1, 1, 1};
    num_blocks_ = {1, 1, 1};
    ::CreateIntegrator<<<num_blocks_, threads_per_block_>>>(pixel_buffer_, texture_buffer_,
                                                            bsdf_buffer_, primitive_buffer_,
                                                            instance_buffer_, accel_,
                                                            num_emitter_, emitter_buffer_,
                                                            num_area_light_, area_light_id_buffer_,
                                                            env_map_, sun_, integrator_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
}

CudaPainter::~CudaPainter()
{
    // 释放相机
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(camera_));

    // 释放 Kulla-Conty LUT
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(brdf_avg_buffer_));
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(albedo_avg_buffer_));

    // 释放纹理
    threads_per_block_ = {32, 1, 1};
    num_blocks_ = {static_cast<unsigned int>(num_texture_ / 32 + 1), 1, 1};
    ::DeleteTextureBuffer<<<num_blocks_, threads_per_block_>>>(num_texture_, texture_buffer_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaFree(texture_buffer_));

    // 释放 BSDF
    threads_per_block_ = {32, 1, 1};
    num_blocks_ = {static_cast<unsigned int>(num_bsdf_ / 32 + 1), 1, 1};
    ::DeleteBsdfBuffer<<<num_blocks_, threads_per_block_>>>(num_bsdf_, bsdf_buffer_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaFree(bsdf_buffer_));

    // 释放图元及用于加速计算的数据结构
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(primitive_buffer_));

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(bvh_node_buffer_));

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(accel_));

    // 释放物体实例及光源信息
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(instance_buffer_));

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(area_light_id_buffer_));

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(env_map_));

    threads_per_block_ = {32, 1, 1};
    num_blocks_ = {static_cast<unsigned int>(num_emitter_ / 32 + 1), 1, 1};
    ::DeleteEmitterBuffer<<<num_blocks_, threads_per_block_>>>(num_emitter_, emitter_buffer_);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());
    CheckCudaErrors(cudaFree(emitter_buffer_));

    // 释放位图纹理引用的像素
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(pixel_buffer_));

    // 释放积分器
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(integrator_));
}

void CudaPainter::Draw(const std::string &filename)
{
    fprintf(stderr, "[info] begin rendering ...\n");
    Timer timer;

    // 创建帧
    float *frame_buffer = nullptr;
    uint32_t num_component = 3 * camera_->width() * camera_->height();
    CheckCudaErrors(cudaMallocManaged((void **)&frame_buffer, num_component * sizeof(float)));

    threads_per_block_ = {8, 8, 1};
    num_blocks_ = {static_cast<unsigned int>(camera_->width() / 8 + 1),
                   static_cast<unsigned int>(camera_->height() / 8 + 1), 1};
    ::DispatchRays<<<num_blocks_, threads_per_block_>>>(camera_, integrator_, frame_buffer);
    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaDeviceSynchronize());

    timer.PrintTimePassed("rendering");
    image_io::Write(camera_->width(), camera_->height(), frame_buffer, filename);

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(frame_buffer));
}

#endif