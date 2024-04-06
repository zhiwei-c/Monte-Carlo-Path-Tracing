#include "csrt/renderer/renderer.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>

#include "csrt/renderer/bsdfs/kulla_conty.hpp"

namespace
{

using namespace csrt;

std::mutex g_mutex_patch;
std::vector<std::vector<std::array<uint32_t, 3>>> g_patches;
#ifdef ENABLE_CUDA
dim3 g_threads_per_block = {8, 8, 1};
dim3 g_num_blocks = {1, 1, 1};
#endif

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
    std::sort(
        pixels.begin(), pixels.end(),
        [](const std::array<uint32_t, 3> &a, const std::array<uint32_t, 3> &b)
        { return a.at(2) < b.at(2); });

    constexpr uint32_t patch_size = 64;
    const uint32_t num_patch = (pixels.size() + patch_size - 1) / patch_size;
    g_patches = std::vector<std::vector<std::array<uint32_t, 3>>>(num_patch);

    uint32_t begin, end;
    for (uint32_t i = 0; i < num_patch; ++i)
    {
        begin = i * patch_size,
        end = std::min((i + 1) * patch_size, resolution);
        g_patches[i] = std::vector<std::array<uint32_t, 3>>(
            pixels.begin() + begin, pixels.begin() + end);
    }
}

QUALIFIER_D_H void DrawPixel(const uint32_t i, const uint32_t j, Camera *camera,
                             Integrator *integrator, float *frame)
{
    const uint32_t pixel_offset = (j * camera->width() + i) * 3;
    uint32_t seed = Tea<4>(pixel_offset, 0);
    Vec3 color, temp;
    for (uint32_t s = 0; s < camera->spp(); ++s)
    {
        const float u = s * camera->spp_inv(),
                    v = GetVanDerCorputSequence<2>(s + 1),
                    x = 2.0f * (i + u) / camera->width() - 1.0f,
                    y = 1.0f - 2.0f * (j + v) / camera->height();
        const Vec3 look_dir = Normalize(
            camera->front() + x * camera->view_dx() + y * camera->view_dy());
        temp = integrator->Shade(camera->eye(), look_dir, &seed);
        temp.x = fminf(temp.x, 1.0f);
        temp.y = fminf(temp.y, 1.0f);
        temp.z = fminf(temp.z, 1.0f);
        color += temp;
    }
    color *= camera->spp_inv();
    for (int channel = 0; channel < 3; ++channel)
        frame[pixel_offset + channel] = color[channel];
}

#ifdef ENABLE_CUDA
__global__ void DispathRaysCuda(Camera *camera, Integrator *integrator,
                                float *frame)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x,
                   j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < camera->width() && j < camera->height())
        DrawPixel(i, j, camera, integrator, frame);
}

__global__ void DispathRaysCuda(Camera *camera, Integrator *integrator,
                                const uint32_t index_frame, float *frame,
                                float *frame_srgb)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x,
                   j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < camera->width() && j < camera->height())
    {
        const float u = GetVanDerCorputSequence<2>(index_frame + 1),
                    v = GetVanDerCorputSequence<3>(index_frame + 1),
                    x = 2.0f * (i + u) / camera->width() - 1.0f,
                    y = 1.0f - 2.0f * (j + v) / camera->height();
        const Vec3 look_dir = Normalize(
            camera->front() + x * camera->view_dx() + y * camera->view_dy());

        const uint32_t pixel_offset = (j * camera->width() + i) * 3,
                       offset_dest =
                           ((camera->height() - 1 - j) * camera->width() + i) *
                           3;
        uint32_t seed = Tea<4>(pixel_offset, index_frame);

        Vec3 color = integrator->Shade(camera->eye(), look_dir, &seed);
        for (int c = 0; c < 3; ++c)
        {
            color[c] = fminf(color[c], 1.0f);
            frame[pixel_offset + c] =
                (index_frame * frame[pixel_offset + c] + color[c]) /
                (index_frame + 1);

            if (frame[pixel_offset + c] <= 0.0031308f)
            {
                frame_srgb[offset_dest + c] = 12.92f * frame[pixel_offset + c];
            }
            else
            {
                frame_srgb[offset_dest + c] =
                    1.055f * powf(frame[pixel_offset + c], 1.0f / 2.4f) -
                    0.055f;
            }
        }
    }
}

#endif

void DispathRaysCpu(Camera *camera, Integrator *integrator, float *frame)
{
    Timer timer;
    uint64_t count_pacth = 0;
    const uint64_t size_patch = g_patches.size();
    const double size_patch_rcp = 1.0 / size_patch;

#if defined(DEBUG) || defined(_DEBUG)

    // {
    //     Vec3 temp1;
    //     for (int k = 0; k < 10; ++k)
    //     {
    //         int i = 643, j = 579;
    //         uint32_t s = 31, seed = 3254056006;
    //         const float u = s * camera->spp_inv(),
    //                     v = GetVanDerCorputSequence<2>(s + 1),
    //                     x = 2.0f * (i + u) / camera->width() - 1.0f,
    //                     y = 1.0f - 2.0f * (j + v) / camera->height();
    //         const Vec3 look_dir =
    //                        Normalize(camera->front() + x * camera->view_dx()
    //                        +
    //                                  y * camera->view_dy()),
    //                    eye = camera->eye();
    //         temp1 = integrator->Shade(eye, look_dir, &seed);
    //     }
    // }

    {
        std::vector<std::array<int, 2>> pixel;

        for (int i = 0; i < 10; ++i)
        {
            pixel.push_back({1026, 710});
            pixel.push_back({1026, 711});
        }

        std::vector<Vec3> ret;
        std::vector<std::vector<Vec3>> colors;
        std::vector<std::vector<uint32_t>> seeds;
        int cnt = 0;
        for (auto [i, j] : pixel)
        {
            const uint32_t pixel_offset = (j * camera->width() + i) * 3;
            uint32_t seed = Tea<4>(pixel_offset, 0);
            Vec3 color;
            colors.push_back(std::vector<Vec3>());
            seeds.push_back(std::vector<uint32_t>());
            for (uint32_t s = 0; s < camera->spp(); ++s)
            {
                seeds[cnt].push_back(seed);
                const float u = s * camera->spp_inv(),
                            v = GetVanDerCorputSequence<2>(s + 1),
                            x = 2.0f * (i + u) / camera->width() - 1.0f,
                            y = 1.0f - 2.0f * (j + v) / camera->height();
                const Vec3 look_dir = Normalize(camera->front() +
                                                x * camera->view_dx() +
                                                y * camera->view_dy()),
                           eye = camera->eye();
                Vec3 temp = integrator->Shade(eye, look_dir, &seed);
                colors[cnt].push_back(temp);
                color += temp;
            }
            cnt++;
            color *= camera->spp_inv();
            for (int channel = 0; channel < 3; ++channel)
            {
                color[channel] =
                    (color[channel] <= 0.0031308f)
                        ? (12.92f * color[channel])
                        : (1.055f * powf(color[channel], 1.0f / 2.4f) - 0.055f);
            }
            ret.push_back(color);
        }
    }

#endif

    auto DispatchRay = [&]()
    {
        uint64_t id_patch = 0;
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(g_mutex_patch);
                if (count_pacth == size_patch)
                    break;
                id_patch = count_pacth++;
            }
            for (const std::array<uint32_t, 3> &pixel : g_patches[id_patch])
            {
                const uint32_t i = pixel.at(0), j = pixel.at(1);
                DrawPixel(i, j, camera, integrator, frame);
            }
            {
                std::lock_guard<std::mutex> lock(g_mutex_patch);
                timer.PrintProgress(count_pacth * size_patch_rcp);
            }
        }
    };

    std::vector<std::thread> workers;
    const unsigned int thread_num = std::thread::hardware_concurrency();
    for (unsigned int i = 0; i < thread_num; ++i)
    {
        workers.push_back(std::thread{DispatchRay});
    }

    for (std::thread &worker : workers)
        worker.join();

    timer.PrintTimePassed("rendering");
}
} // namespace

namespace csrt
{

Renderer::Renderer(const RendererConfig &config)
    : backend_type_(config.backend_type), camera_(nullptr), textures_(nullptr),
      bsdfs_(nullptr), media_(nullptr), emitters_(nullptr),
      integrator_(nullptr), map_instance_bsdf_(nullptr),
      map_area_light_instance_(nullptr), map_instance_area_light_(nullptr),
      cdf_area_light_(nullptr), pixels_(nullptr), data_env_map_(nullptr),
      brdf_avg_buffer_(nullptr), albedo_avg_buffer_(nullptr)
{
    try
    {
        scene_ = new csrt::Scene(config.backend_type, config.instances);

        const size_t num_instance = config.instances.size();
        std::vector<uint32_t> map_area_light_instance;
        std::vector<float> list_area_light_weight;
        map_instance_bsdf_ = MallocArray<uint32_t>(backend_type_, num_instance);
        for (size_t i = 0; i < num_instance; ++i)
        {
            map_instance_bsdf_[i] = config.instances[i].id_bsdf;
            if (config.instances[i].id_bsdf < config.bsdfs.size())
            {
                const csrt::BsdfInfo info_bsdf =
                    config.bsdfs[config.instances[i].id_bsdf];
                if (info_bsdf.type == csrt::BsdfType::kAreaLight)
                {
                    map_area_light_instance.push_back(i);
                    list_area_light_weight.push_back(
                        info_bsdf.area_light.weight);
                }
            }
        }
        map_area_light_instance_ =
            MallocArray<uint32_t>(backend_type_, map_area_light_instance);

        map_instance_area_light_ = MallocArray<uint32_t>(
            backend_type_, std::vector<uint32_t>(num_instance, kInvalidId));
        const uint32_t num_area_light =
            static_cast<uint32_t>(list_area_light_weight.size());
        cdf_area_light_ = MallocArray<float>(backend_type_, num_area_light + 1);
        cdf_area_light_[0] = 0;
        for (uint32_t i = 0; i < num_area_light; ++i)
        {
            cdf_area_light_[i + 1] =
                list_area_light_weight[i] + cdf_area_light_[i];
            map_instance_area_light_[map_area_light_instance[i]] = i;
        }

        camera_ = MallocElement<Camera>(backend_type_);
        *camera_ = Camera(config.camera);

        CommitTextures(config.textures);

        brdf_avg_buffer_ =
            MallocArray<float>(backend_type_, kLutResolution * kLutResolution);
        albedo_avg_buffer_ = MallocArray<float>(backend_type_, kLutResolution);
        ComputeKullaConty(brdf_avg_buffer_, albedo_avg_buffer_);

        CommitBsdfs(config.textures.size(), config.bsdfs);
        CommitMedia(config.media);

        uint32_t id_sun = kInvalidId, id_envmap = kInvalidId;
        CommitEmitters(config.textures, config.emitters, &id_sun, &id_envmap);

        CommitIntegrator(config.integrator, num_area_light,
                         static_cast<uint32_t>(config.emitters.size()), id_sun,
                         id_envmap);

#ifdef ENABLE_CUDA
        if (backend_type_ == BackendType::kCpu)
        {
#endif
            GeneratePatchInfo(camera_->width(), camera_->height());
#ifdef ENABLE_CUDA
        }
        else
        {
            g_num_blocks = {
                static_cast<unsigned int>(camera_->width() / 8 + 1),
                static_cast<unsigned int>(camera_->height() / 8 + 1), 1};
        }
#endif
    }
    catch (const std::exception &e)
    {
        ReleaseData();
        std::ostringstream oss;
        oss << "error when commit renderer.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Renderer::ReleaseData()
{
    DeleteElement(BackendType::kCpu, scene_);

    DeleteElement(backend_type_, camera_);
    DeleteArray(backend_type_, textures_);
    DeleteArray(backend_type_, bsdfs_);
    DeleteArray(backend_type_, media_);
    DeleteArray(backend_type_, emitters_);
    DeleteElement(backend_type_, integrator_);

    DeleteArray(backend_type_, map_instance_bsdf_);
    DeleteArray(backend_type_, map_area_light_instance_);
    DeleteArray(backend_type_, map_instance_area_light_);
    DeleteArray(backend_type_, cdf_area_light_);
    DeleteArray(backend_type_, pixels_);
    DeleteArray(backend_type_, data_env_map_);
    DeleteArray(backend_type_, brdf_avg_buffer_);
    DeleteArray(backend_type_, albedo_avg_buffer_);
}

void Renderer::CommitTextures(const std::vector<TextureInfo> &list_texture_info)
{
    std::vector<float> pixel_buffer;
    std::vector<uint64_t> offsets;
    uint64_t accumulate_pixel_num = 0;
    for (const TextureInfo &info : list_texture_info)
    {
        if (info.type == TextureType::kBitmap)
        {
            pixel_buffer.insert(pixel_buffer.end(), info.bitmap.data.begin(),
                                info.bitmap.data.end());
            offsets.push_back(accumulate_pixel_num);

            const int pixel_num =
                info.bitmap.width * info.bitmap.height * info.bitmap.channel;
            accumulate_pixel_num += pixel_num;
        }
    }

    try
    {
        pixels_ = MallocArray(backend_type_, pixel_buffer);

        const size_t num_texture = list_texture_info.size();
        textures_ = MallocArray<Texture>(backend_type_, num_texture);

        size_t data_offset = 0;
        for (size_t i = 0, j = 0; i < num_texture; ++i)
        {
            TextureData data;
            data.type = list_texture_info[i].type;
            switch (data.type)
            {
            case TextureType::kConstant:
                data.constant = list_texture_info[i].constant;
                break;
            case TextureType::kCheckerboard:
                data.checkerboard = list_texture_info[i].checkerboard;
                break;
            case TextureType::kBitmap:
                data.bitmap.width = list_texture_info[i].bitmap.width;
                data.bitmap.height = list_texture_info[i].bitmap.height;
                data.bitmap.channel = list_texture_info[i].bitmap.channel;
                data.bitmap.to_uv = list_texture_info[i].bitmap.to_uv;
                data.bitmap.data = pixels_;
                data_offset = offsets[j++];
                break;
            default:
                throw MyException("unknow texture type.");
                break;
            }
            textures_[i] = Texture(i, data, data_offset);
        }
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when commit textures to renderer.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Renderer::CommitBsdfs(const size_t num_texture,
                           const std::vector<BsdfInfo> &list_bsdf_info)
{
    auto CheckTexture = [&](const uint32_t id, const bool allow_invalid)
    {
        if (id == kInvalidId && allow_invalid)
            return;

        if (id >= num_texture)
        {
            std::ostringstream oss;
            oss << "cannot find texture (id " << id << ").";
            throw MyException(oss.str());
        }
    };

    try
    {
        const size_t num_bsdf = list_bsdf_info.size();
        bsdfs_ = MallocArray<Bsdf>(backend_type_, num_bsdf);
        for (size_t i = 0; i < num_bsdf; ++i)
        {
            const BsdfInfo &info = list_bsdf_info[i];
            CheckTexture(info.id_opacity, true);
            CheckTexture(info.id_bump_map, true);
            switch (info.type)
            {
            case BsdfType::kAreaLight:
                CheckTexture(info.area_light.id_radiance, false);
                break;
            case BsdfType::kDiffuse:
                CheckTexture(info.diffuse.id_diffuse_reflectance, false);
                break;
            case BsdfType::kRoughDiffuse:
                CheckTexture(info.rough_diffuse.id_diffuse_reflectance, false);
                CheckTexture(info.rough_diffuse.id_roughness, false);
                break;
            case BsdfType::kConductor:
                CheckTexture(info.conductor.id_roughness_u, false);
                CheckTexture(info.conductor.id_roughness_v, false);
                CheckTexture(info.conductor.id_specular_reflectance, false);
                break;
            case BsdfType::kThinDielectric:
            case BsdfType::kDielectric:
                CheckTexture(info.dielectric.id_roughness_u, false);
                CheckTexture(info.dielectric.id_roughness_v, false);
                CheckTexture(info.dielectric.id_specular_reflectance, false);
                CheckTexture(info.dielectric.id_specular_transmittance, false);
                break;
            case BsdfType::kPlastic:
                CheckTexture(info.plastic.id_roughness, false);
                CheckTexture(info.plastic.id_diffuse_reflectance, false);
                CheckTexture(info.plastic.id_specular_reflectance, false);
                break;
            default:
                throw MyException("unknow BSDF type.");
                break;
            }
            bsdfs_[i] =
                Bsdf(i, info, textures_, brdf_avg_buffer_, albedo_avg_buffer_);
        }
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when commit BSDFs to renderer." << e.what();
        throw MyException(oss.str());
    }
}

void Renderer::CommitMedia(const std::vector<MediumInfo> &list_medium_info)
{
    try
    {
        const size_t num_medium = list_medium_info.size();
        media_ = MallocArray<Medium>(backend_type_, num_medium);
        for (size_t i = 0; i < num_medium; ++i)
        {
            media_[i] = Medium(i, list_medium_info[i]);
        }
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when commit medium to renderer." << e.what();
        throw MyException(oss.str());
    }
}

void Renderer::CommitEmitters(const std::vector<TextureInfo> &list_texture_info,
                              const std::vector<EmitterInfo> &list_emitter_info,
                              uint32_t *id_sun, uint32_t *id_envmap)
{
    auto CheckTexture = [&](const uint32_t id, const bool allow_invalid)
    {
        if (id == kInvalidId && allow_invalid)
            return;

        if (id >= list_texture_info.size())
        {
            std::ostringstream oss;
            oss << "cannot find texture (id " << id << ").";
            throw MyException(oss.str());
        }
    };

    try
    {
        *id_sun = kInvalidId;
        *id_envmap = kInvalidId;

        const uint64_t num_emitter = list_emitter_info.size();
        emitters_ = MallocArray<Emitter>(backend_type_, num_emitter);

        for (uint64_t i = 0; i < num_emitter; ++i)
        {
            const EmitterInfo &info = list_emitter_info[i];
            switch (info.type)
            {
            case EmitterType::kPoint:
            case EmitterType::kSpot:
            case EmitterType::kDirectional:
                break;
            case EmitterType::kSun:
                CheckTexture(info.sun.id_texture, false);
                *id_sun = i;
                break;
            case EmitterType::kEnvMap:
                CheckTexture(info.envmap.id_radiance, false);
            case EmitterType::kConstant:
                *id_envmap = i;
                break;
            default:
                throw MyException("unknow emitter type.");
                break;
            }
            emitters_[i] = Emitter(i, info, textures_);

            if (info.type == EmitterType::kEnvMap)
            {
                const TextureInfo &radiance_texture_info =
                    list_texture_info[info.envmap.id_radiance];
                if (radiance_texture_info.type != TextureType::kBitmap)
                {
                    std::ostringstream oss;
                    oss << "radiance texture '" << info.envmap.id_radiance
                        << "' for emitter '" << i << "' is not a bitmap.";
                    throw MyException(oss.str());
                }

                std::vector<float> cdf_cols, cdf_rows, weight_rows;
                float normalization;

                CreateEnvMapCdfPdf(radiance_texture_info.bitmap.width,
                                   radiance_texture_info.bitmap.height,
                                   textures_[info.envmap.id_radiance],
                                   &cdf_cols, &cdf_rows, &weight_rows,
                                   &normalization);

                DeleteArray(backend_type_, data_env_map_);
                const uint64_t num_data =
                    cdf_cols.size() + cdf_rows.size() + weight_rows.size();
                data_env_map_ = MallocArray<float>(backend_type_, num_data);

                for (uint64_t j = 0; j < cdf_rows.size(); ++j)
                    data_env_map_[j] = cdf_rows[j];

                uint64_t offset = cdf_rows.size();
                for (uint64_t j = 0; j < weight_rows.size(); ++j)
                    data_env_map_[j + offset] = weight_rows[j];

                offset += weight_rows.size();
                for (uint64_t j = 0; j < cdf_cols.size(); ++j)
                    data_env_map_[j + offset] = cdf_cols[j];

                emitters_[i].InitEnvMap(radiance_texture_info.bitmap.width,
                                        radiance_texture_info.bitmap.height,
                                        normalization, data_env_map_);
            }
        }
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when commit BSDFs to renderer." << e.what();
        throw MyException(oss.str());
    }
}

void Renderer::CommitIntegrator(const IntegratorInfo &integrator_info,
                                const uint32_t num_area_light,
                                const uint32_t num_emitter,
                                const uint32_t id_sun, const uint32_t id_envmap)
{
    try
    {
        integrator_ = MallocElement<Integrator>(backend_type_);

        IntegratorData data_integrator;
        data_integrator.info = integrator_info;
        data_integrator.size_cdf_area_light = num_area_light + 1;
        data_integrator.pdf_rr_rcp = integrator_info.pdf_rr;

        data_integrator.num_area_light = num_area_light;
        data_integrator.num_emitter = num_emitter;
        data_integrator.id_sun = id_sun;
        data_integrator.id_envmap = id_envmap;

        data_integrator.bsdfs = bsdfs_;

        data_integrator.instances = scene_->GetInstances();
        data_integrator.list_pdf_area_instance = scene_->GetPdfAreaList();

        data_integrator.emitters = emitters_;
        data_integrator.map_id_area_light_instance = map_area_light_instance_;
        data_integrator.map_id_instance_area_light = map_instance_area_light_;
        data_integrator.cdf_area_light = cdf_area_light_;

        data_integrator.tlas = scene_->GetTlas();
        data_integrator.map_instance_bsdf = map_instance_bsdf_;

        switch (integrator_info.type)
        {
        case IntegratorType::kPath:
            data_integrator.info.type = IntegratorType::kPath;
            break;
        case IntegratorType::kVolPath:
            data_integrator.info.type = IntegratorType::kVolPath;
            data_integrator.media = media_;
            break;
        default:
            throw MyException("unknow integrator type");
            break;
        }

        *integrator_ = Integrator(data_integrator);
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when set integrator to renderer.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

void Renderer::Draw(float *frame) const
{
    try
    {
        fprintf(stderr, "[info] begin rendering ...\n");

#ifdef ENABLE_CUDA
        if (backend_type_ == BackendType::kCpu)
        {
#endif
            DispathRaysCpu(camera_, integrator_, frame);
#ifdef ENABLE_CUDA
        }
        else
        {
            Timer timer;
            DispathRaysCuda<<<g_num_blocks, g_threads_per_block>>>(
                camera_, integrator_, frame);
            cudaError_t ret = cudaGetLastError();
            if (ret)
            {
                std::ostringstream oss;
                oss << "CUDA error : \"" << ret << "\".";
                throw MyException(oss.str());
            }

            ret = cudaDeviceSynchronize();
            if (ret)
            {
                std::ostringstream oss;
                oss << "CUDA error : \"" << ret << "\".";
                throw MyException(oss.str());
            }
            timer.PrintTimePassed("rendering");
        }
#endif
    }
    catch (const MyException &e)
    {
        std::ostringstream oss;
        oss << "error when draw.\n\t" << e.what();
        throw MyException(oss.str());
    }
}

#ifdef ENABLE_VIEWER
void Renderer::Draw(const uint32_t index_frame, float *frame,
                    float *frame_srgb) const
{
    DispathRaysCuda<<<g_num_blocks, g_threads_per_block>>>(
        camera_, integrator_, index_frame, frame, frame_srgb);

    cudaError_t ret = cudaGetLastError();
    if (ret)
    {
        std::ostringstream oss;
        oss << "CUDA error : \"" << ret << "\" when draw.";
        throw MyException(oss.str());
    }

    ret = cudaDeviceSynchronize();
    if (ret)
    {
        std::ostringstream oss;
        oss << "CUDA error : \"" << ret << "\" when draw.";
        throw MyException(oss.str());
    }
}
#endif

} // namespace csrt