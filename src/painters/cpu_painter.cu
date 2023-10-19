#include "painters.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>
#include <array>
#include <thread>
#include <mutex>

#include "../utils/math.cuh"
#include "../utils/misc.cuh"
#include "../utils/image_io.cuh"
#include "../utils/timer.cuh"

namespace cpu
{
    std::mutex mutex_patch;
    std::vector<std::vector<std::array<uint32_t, 3>>> patches;

    void GeneratePatchInfo(const uint32_t width, const uint32_t height)
    {
        const uint32_t resolution = width * height;
        std::vector<std::array<uint32_t, 3>> pixels(resolution);
        auto GetMortonCode = [](const uint32_t column, const uint32_t row)
        {
            uint32_t morton = 0;
            for (uint32_t i = 0; i < sizeof(row) * 8; ++i)
            {
                morton |= ((row & static_cast<uint32_t>(1) << i) << i |
                           (column & static_cast<uint32_t>(1) << i) << (i + 1));
            }
            return morton;
        };
        for (uint32_t j = 0; j < height; ++j)
        {
            for (uint32_t i = 0; i < width; ++i)
                pixels[j * width + i] = {i, j, GetMortonCode(i, j)};
        }
        std::sort(pixels.begin(), pixels.end(),
                  [](const std::array<uint32_t, 3> &a, const std::array<uint32_t, 3> &b)
                  { return a.at(2) < b.at(2); });

        constexpr uint32_t patch_size = 64;
        const uint32_t num_patch = (pixels.size() + patch_size - 1) / patch_size;
        ::cpu::patches = std::vector<std::vector<std::array<uint32_t, 3>>>(num_patch);

        uint32_t begin, end;
        for (uint32_t i = 0; i < num_patch; ++i)
        {
            begin = i * patch_size,
            end = std::min((i + 1) * patch_size, resolution);
            ::cpu::patches[i] = std::vector<std::array<uint32_t, 3>>(
                pixels.begin() + begin, pixels.begin() + end);
        }
    }

    void DispatchRays(Integrator *integrator, Camera *camera, uint32_t &count_pacth,
                      Timer &timer, float *frame_buffer)
    {
        uint32_t id_patch = 0;
        const double one_div_num_patch = 1.0 / patches.size();
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(::cpu::mutex_patch);
                if (count_pacth == patches.size())
                    break;
                id_patch = count_pacth++;
            }
            for (const std::array<uint32_t, 3> &pixel : patches[id_patch])
            {
                const uint32_t i = pixel.at(0),
                               j = pixel.at(1),
                               pixel_index = (j * camera->width() + i) * 3;
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
                    frame_buffer[pixel_index + channel] =
                        (color[channel] <= 0.0031308f)
                            ? (12.92f * color[channel])
                            : (1.055f * powf(color[channel], 1.0f / 2.4f) - 0.055f);
                }
            }
            {
                std::lock_guard<std::mutex> lock(::cpu::mutex_patch);
                timer.PrintProgress(count_pacth * one_div_num_patch);
            }
        }
    }
} // namespace

CpuPainter::CpuPainter(BvhBuilder::Type bvh_type, const SceneInfo &info)
{
    fprintf(stderr, "[info] setup scene ...\n");

    // 创建相机
    camera_ = new Camera(info.camera);

    // 创建纹理
    num_texture_ = info.texture_info_buffer.size();
    texture_buffer_ = nullptr;
    if (!info.texture_info_buffer.empty())
    {
        texture_buffer_ = new Texture *[num_texture_];
        for (uint32_t i = 0; i < num_texture_; ++i)
        {
            switch (info.texture_info_buffer[i].type)
            {
            case Texture::Type::kConstant:
                texture_buffer_[i] = new ConstantTexture(
                    i, info.texture_info_buffer[i].data.constant);
                break;
            case Texture::Type::kCheckerboard:
                texture_buffer_[i] = new CheckerboardTexture(
                    i, info.texture_info_buffer[i].data.checkerboard);
                break;
            case Texture::Type::kBitmap:
                texture_buffer_[i] = new Bitmap(i, info.texture_info_buffer[i].data.bitmap);
                break;
            }
        }
    }

    // 创建 Kulla-Conty LUT
    brdf_avg_buffer_ = new float[Bsdf::kLutResolution * Bsdf::kLutResolution],
    albedo_avg_buffer_ = new float[Bsdf::kLutResolution];
    Bsdf::ComputeKullaConty(brdf_avg_buffer_, albedo_avg_buffer_);

    std::vector<std::vector<float>> brdf_avg_buffer(Bsdf::kLutResolution, std::vector<float>(Bsdf::kLutResolution));
    std::vector<float> albedo_avg_buffer(Bsdf::kLutResolution);
    for (uint32_t i = 0; i < Bsdf::kLutResolution; ++i)
    {
        albedo_avg_buffer[i] = albedo_avg_buffer_[i];
        for (uint32_t j = 0; j < Bsdf::kLutResolution; ++j)
            brdf_avg_buffer[i][j] = brdf_avg_buffer_[i * Bsdf::kLutResolution + j];
    }

    // 创建 BSDF
    num_bsdf_ = info.bsdf_info_buffer.size();
    bsdf_buffer_ = nullptr;
    if (!info.bsdf_info_buffer.empty())
    {
        bsdf_buffer_ = new Bsdf *[num_bsdf_];
        for (uint32_t i = 0; i < num_bsdf_; ++i)
        {
            switch (info.bsdf_info_buffer[i].type)
            {
            case Bsdf::Type::kAreaLight:
                bsdf_buffer_[i] = new AreaLight(i, info.bsdf_info_buffer[i].data);
                break;
            case Bsdf::Type::kDiffuse:
                bsdf_buffer_[i] = new Diffuse(i, info.bsdf_info_buffer[i].data);
                break;
            case Bsdf::Type::kRoughDiffuse:
                bsdf_buffer_[i] = new RoughDiffuse(i, info.bsdf_info_buffer[i].data);
                break;
            case Bsdf::Type::kConductor:
                bsdf_buffer_[i] = new Conductor(i, info.bsdf_info_buffer[i].data);
                break;
            case Bsdf::Type::kDielectric:
                bsdf_buffer_[i] = new Dielectric(i, info.bsdf_info_buffer[i].data);
                break;
            case Bsdf::Type::kThinDielectric:
                bsdf_buffer_[i] = new ThinDielectric(i, info.bsdf_info_buffer[i].data);
                break;
            case Bsdf::Type::kPlastic:
                bsdf_buffer_[i] = new Plastic(i, info.bsdf_info_buffer[i].data);
                break;
            }
            bsdf_buffer_[i]->SetKullaConty(brdf_avg_buffer_, albedo_avg_buffer_);
        }
    }

    // 创建图元
    num_primitive_ = info.primitive_info_buffer.size();
    fprintf(stderr, "[info] total primitive num : %lu\n", num_primitive_);

    primitive_buffer_ = nullptr;
    std::vector<AABB> aabb_buffer(num_primitive_);
    if (num_primitive_ > 0)
    {
        primitive_buffer_ = new Primitive[num_primitive_];
        for (uint32_t i = 0; i < num_primitive_; ++i)
        {
            primitive_buffer_[i] = Primitive(i, info.primitive_info_buffer[i]);
            aabb_buffer[i] = primitive_buffer_[i].aabb();
        }
    }

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

        bvh_node_buffer_ = new BvhNode[bvh_node_buffer.size()];
        std::copy(bvh_node_buffer.begin(), bvh_node_buffer.end(), bvh_node_buffer_);
        accel_ = new Accel(primitive_buffer_, bvh_node_buffer_);
    }

    // 创建物体实例，处理面光源信息
    instance_buffer_ = nullptr;
    std::vector<uint32_t> area_light_id_buffer;
    if (!info.instance_info_buffer.empty())
    {
        uint32_t num_instance = info.instance_info_buffer.size();
        instance_buffer_ = new Instance[num_instance];
        for (uint32_t i = 0; i < num_instance; ++i)
        {
            instance_buffer_[i] = Instance(i, info.instance_info_buffer[i], primitive_buffer_);

            if (info.instance_info_buffer[i].is_emitter)
                area_light_id_buffer.push_back(i);
        }
    }
    num_area_light_ = area_light_id_buffer.size();
    area_light_id_buffer_ = nullptr;
    if (!area_light_id_buffer.empty())
    {
        area_light_id_buffer_ = new uint32_t[num_area_light_];
        std::copy(area_light_id_buffer.begin(), area_light_id_buffer.end(), area_light_id_buffer_);
    }

    //
    // 创建除面光源之外的其它光源
    //
    env_map_ = nullptr;
    if (info.env_map)
        env_map_ = new EnvMap(*info.env_map);

    num_emitter_ = info.emitter_info_buffer.size();
    emitter_buffer_ = nullptr;
    sun_ = nullptr;
    if (!info.emitter_info_buffer.empty())
    {
        emitter_buffer_ = new Emitter *[num_emitter_];
        for (uint32_t i = 0; i < num_emitter_; ++i)
        {
            switch (info.emitter_info_buffer[i].type)
            {
            case Emitter::Type::kSpot:
                emitter_buffer_[i] = new SpotLight(i, info.emitter_info_buffer[i].data.spot);
                break;
            case Emitter::Type::kDirectional:
                emitter_buffer_[i] = new DirectionalLight(
                    i, info.emitter_info_buffer[i].data.directional);
                break;
            case Emitter::Type::kSun:
                emitter_buffer_[i] = new Sun(i, info.emitter_info_buffer[i].data.sun);
                sun_ = (Sun *)emitter_buffer_[i];
                break;
            }
        }
    }

    // 创建位图纹理引用的像素
    pixel_buffer_ = nullptr;
    if (!info.pixel_buffer.empty())
    {
        pixel_buffer_ = new float[info.pixel_buffer.size()];
        std::copy(info.pixel_buffer.begin(), info.pixel_buffer.end(), pixel_buffer_);
    }

    // 创建积分器
    integrator_ = new Integrator(pixel_buffer_, texture_buffer_, bsdf_buffer_, primitive_buffer_,
                                 instance_buffer_, accel_, num_emitter_, emitter_buffer_,
                                 num_area_light_, area_light_id_buffer_, env_map_, sun_);
}

CpuPainter::~CpuPainter()
{
    SAFE_DELETE_ELEMENT(camera_);

    SAFE_DELETE_ARRAY(brdf_avg_buffer_);
    SAFE_DELETE_ARRAY(albedo_avg_buffer_);

    SAFE_DELETE_ARRAY(pixel_buffer_);
    for (uint32_t i = 0; i < num_texture_; ++i)
        SAFE_DELETE_ELEMENT(texture_buffer_[i]);

    SAFE_DELETE_ARRAY(texture_buffer_);

    for (uint32_t i = 0; i < num_bsdf_; ++i)
        SAFE_DELETE_ELEMENT(bsdf_buffer_[i]);
    SAFE_DELETE_ARRAY(bsdf_buffer_);

    SAFE_DELETE_ARRAY(primitive_buffer_);
    SAFE_DELETE_ARRAY(instance_buffer_);

    for (uint32_t i = 0; i < num_emitter_; ++i)
        SAFE_DELETE_ELEMENT(emitter_buffer_[i]);
    SAFE_DELETE_ARRAY(emitter_buffer_);

    SAFE_DELETE_ARRAY(area_light_id_buffer_);
    SAFE_DELETE_ELEMENT(env_map_);

    SAFE_DELETE_ARRAY(bvh_node_buffer_);

    SAFE_DELETE_ELEMENT(accel_);

    SAFE_DELETE_ELEMENT(integrator_);
}

void CpuPainter::Draw(const std::string &filename)
{
    uint32_t count_pacth = 0;
    float *frame_buffer = new float[camera_->width() * camera_->height() * 3];
    ::cpu::GeneratePatchInfo(camera_->width(), camera_->height());

#if defined(DEBUG) || defined(_DEBUG)
    {
        std::vector<Vec3> colors_tmp, ret;
        {
            int i = 305,
                j = 474;
            const uint32_t pixel_index = (j * camera_->width() + i) * 3;
            uint32_t seed = Tea(pixel_index, 0, 4);
            Vec3 color, color_tmp;
            for (uint32_t s = 0; s < camera_->spp(); ++s)
            {
                const float u = s * camera_->spp_inv(),
                            v = GetVanDerCorputSequence(s + 1, 2),
                            x = 2.0f * (i + u) / camera_->width() - 1.0f,
                            y = 1.0f - 2.0f * (j + v) / camera_->height();
                const Vec3 look_dir = Normalize(camera_->front() + x * camera_->view_dx() +
                                                y * camera_->view_dy());
                const Vec3 eye = camera_->eye();
                color_tmp = integrator_->GenerateRay(eye, look_dir, &seed);
                colors_tmp.push_back(color_tmp);
                color += color_tmp;
            }
            color *= camera_->spp_inv();
            for (int channel = 0; channel < 3; ++channel)
            {
                color[channel] = (color[channel] <= 0.0031308f)
                                     ? (12.92f * color[channel])
                                     : (1.055f * powf(color[channel], 1.0f / 2.4f) - 0.055f);
            }
            ret.push_back(color);
        }
    }

#endif

    fprintf(stderr, "[info] begin rendering ...\n");
    Timer timer;

    std::vector<std::thread> workers;
    const unsigned int thread_num = std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < thread_num; ++i)
    {
        workers.push_back(std::thread{::cpu::DispatchRays, integrator_, camera_,
                                      std::ref(count_pacth), std::ref(timer), frame_buffer});
    }

    for (std::thread &worker : workers)
        worker.join();
    timer.PrintTimePassed("rendering");

    image_io::Write(camera_->width(), camera_->height(), frame_buffer, filename);

    SAFE_DELETE_ARRAY(frame_buffer);
}