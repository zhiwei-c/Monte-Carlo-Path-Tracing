#include "csrt/renderer/render.cuh"

#include <algorithm>
#include <array>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>

namespace
{

using namespace csrt;

std::mutex m_mutex_patch;
std::vector<std::vector<std::array<uint32_t, 3>>> m_patches;
#ifdef ENABLE_CUDA
dim3 m_threads_per_block = {8, 8, 1};
dim3 m_num_blocks = {1, 1, 1};
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
    m_patches = std::vector<std::vector<std::array<uint32_t, 3>>>(num_patch);

    uint32_t begin, end;
    for (uint32_t i = 0; i < num_patch; ++i)
    {
        begin = i * patch_size,
        end = std::min((i + 1) * patch_size, resolution);
        m_patches[i] = std::vector<std::array<uint32_t, 3>>(
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
    {
        if (color[channel] <= 0.0031308f)
        {
            frame[pixel_offset + channel] = 12.92f * color[channel];
        }
        else
        {
            frame[pixel_offset + channel] =
                1.055f * powf(color[channel], 1.0f / 2.4f) - 0.055f;
        }
    }
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
                                const uint32_t index_frame, float *accum,
                                float *frame)
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
            accum[pixel_offset + c] =
                (index_frame * accum[pixel_offset + c] + color[c]) /
                (index_frame + 1);

            if (accum[pixel_offset + c] <= 0.0031308f)
            {
                frame[offset_dest + c] = 12.92f * accum[pixel_offset + c];
            }
            else
            {
                frame[offset_dest + c] =
                    1.055f * powf(accum[pixel_offset + c], 1.0f / 2.4f) -
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
    const uint64_t size_patch = m_patches.size();
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
            pixel.push_back({137, 551});

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
                color[channel] = fminf(color[channel], 1.0f);
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
                std::lock_guard<std::mutex> lock(m_mutex_patch);
                if (count_pacth == size_patch)
                    break;
                id_patch = count_pacth++;
            }
            for (const std::array<uint32_t, 3> &pixel : m_patches[id_patch])
            {
                const uint32_t i = pixel.at(0), j = pixel.at(1);
                DrawPixel(i, j, camera, integrator, frame);
            }
            {
                std::lock_guard<std::mutex> lock(m_mutex_patch);
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

Renderer::Renderer(const BackendType backend_type)
    : backend_type_(backend_type), num_instance_(0), num_area_light_(0),
      id_sun_(kInvalidId), id_envmap_(kInvalidId), pixels_(nullptr),
      textures_(nullptr), bsdfs_(nullptr), instances_(nullptr),
      emitters_(nullptr), data_env_map_(nullptr), tlas_(nullptr),
      integrator_(nullptr), camera_(nullptr), cdf_area_light_(nullptr),
      list_pdf_area_instance_(nullptr), map_instance_bsdf_(nullptr),
      map_instance_area_light_(nullptr), map_area_light_instance_(nullptr)
{
}

Renderer::~Renderer()
{
    DeleteArray(backend_type_, textures_);
    DeleteArray(backend_type_, pixels_);
    DeleteElement(backend_type_, camera_);
    DeleteElement(backend_type_, integrator_);

    DeleteArray(backend_type_, bsdfs_);
    DeleteArray(backend_type_, map_instance_bsdf_);

    DeleteArray(backend_type_, map_area_light_instance_);
    DeleteArray(backend_type_, map_instance_area_light_);
    DeleteArray(backend_type_, cdf_area_light_);
    DeleteArray(backend_type_, data_env_map_);
}

void Renderer::AddTexture(const Texture::Info &info)
{
    switch (info.type)
    {
    case Texture::Type::kConstant:
    case Texture::Type::kCheckerboard:
    case Texture::Type::kBitmap1:
    case Texture::Type::kBitmap3:
    case Texture::Type::kBitmap4:
        break;
    default:
        throw std::exception("unknow texture type.");
        break;
    }

    list_texture_info_.push_back(info);
}

void Renderer::AddBsdf(const BSDF::Info &info)
{
    switch (info.type)
    {
    case BSDF::Type::kAreaLight:
    case BSDF::Type::kDiffuse:
    case BSDF::Type::kRoughDiffuse:
    case BSDF::Type::kConductor:
    case BSDF::Type::kDielectric:
    case BSDF::Type::kThinDielectric:
    case BSDF::Type::kPlastic:
        break;
    default:
        throw std::exception("unknow BSDF type.");
        break;
    }

    list_bsdf_info_.push_back(info);
}

void Renderer::AddSceneInfo(Instance *instances, float *list_pdf_area_instance,
                            const std::vector<uint32_t> &map_instance_bsdf,
                            TLAS *tlas)
{
    try
    {
        instances_ = instances;
        list_pdf_area_instance_ = list_pdf_area_instance;
        tlas_ = tlas;

        num_instance_ = map_instance_bsdf.size();
        for (uint32_t i = 0; i < num_instance_; ++i)
            CheckBsdf(map_instance_bsdf[i], true);
        DeleteArray(backend_type_, map_instance_bsdf_);
        map_instance_bsdf_ =
            MallocArray<uint32_t>(backend_type_, map_instance_bsdf);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add scene info to renderer.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::AddEmitter(const Emitter::Info &info)
{
    switch (info.type)
    {
    case Emitter::Type::kPoint:
    case Emitter::Type::kSpot:
    case Emitter::Type::kDirectional:
    case Emitter::Type::kSun:
    case Emitter::Type::kEnvMap:
    case Emitter::Type::kConstant:
        break;
    default:
        throw std::exception("unknow Emitter type.");
        break;
    }

    list_emitter_info_.push_back(info);
}

void Renderer::SetAreaLightInfo(
    const std::vector<uint32_t> map_id_area_light_instance,
    const std::vector<float> list_area_light_weight)
{
    try
    {
        DeleteArray(backend_type_, map_area_light_instance_);
        map_area_light_instance_ =
            MallocArray<uint32_t>(backend_type_, map_id_area_light_instance);

        DeleteArray(backend_type_, map_instance_area_light_);
        map_instance_area_light_ =
            MallocArray<uint32_t>(backend_type_, num_instance_);
        for (uint32_t i = 0; i < num_instance_; ++i)
            map_instance_area_light_[i] = kInvalidId;

        DeleteArray(backend_type_, cdf_area_light_);
        num_area_light_ = static_cast<uint32_t>(list_area_light_weight.size());
        cdf_area_light_ =
            MallocArray<float>(backend_type_, num_area_light_ + 1);
        cdf_area_light_[0] = 0;
        for (uint32_t i = 0; i < num_area_light_; ++i)
        {
            cdf_area_light_[i + 1] =
                list_area_light_weight[i] + cdf_area_light_[i];
            map_instance_area_light_[map_id_area_light_instance[i]] = i;
        }
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when set 'area light' info to renderer.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::SetCamera(const Camera::Info &info)
{
    try
    {
        DeleteElement(backend_type_, camera_);
        camera_ = MallocElement<Camera>(backend_type_);
        *camera_ = Camera(info);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when set camera info to renderer.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::SetIntegrator(const Integrator::Info &info)
{
    info_integrator_ = info;
}

void Renderer::Commit()
{
    try
    {
        CommitTextures();
        CommitBsdfs();
        CommitEmitters();
        CommitIntegrator();
#ifdef ENABLE_CUDA
        if (backend_type_ == BackendType::kCpu)
        {
#endif
            GeneratePatchInfo(camera_->width(), camera_->height());
#ifdef ENABLE_CUDA
        }
        else
        {
            m_num_blocks = {
                static_cast<unsigned int>(camera_->width() / 8 + 1),
                static_cast<unsigned int>(camera_->height() / 8 + 1), 1};
        }
#endif
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit renderer.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
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
            DispathRaysCuda<<<m_num_blocks, m_threads_per_block>>>(
                camera_, integrator_, frame);
            cudaError_t ret = cudaGetLastError();
            if (ret)
            {
                std::ostringstream oss;
                oss << "CUDA error : \"" << ret << "\".";
                throw std::exception(oss.str().c_str());
            }

            ret = cudaDeviceSynchronize();
            if (ret)
            {
                std::ostringstream oss;
                oss << "CUDA error : \"" << ret << "\".";
                throw std::exception(oss.str().c_str());
            }
            timer.PrintTimePassed("rendering");
        }
#endif
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when draw.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

#ifdef ENABLE_VIEWER
void Renderer::Draw(const uint32_t index_frame, float *accum,
                    float *frame) const
{
    DispathRaysCuda<<<m_num_blocks, m_threads_per_block>>>(
        camera_, integrator_, index_frame, accum, frame);

    cudaError_t ret = cudaGetLastError();
    if (ret)
    {
        std::ostringstream oss;
        oss << "CUDA error : \"" << ret << "\" when draw.";
        throw std::exception(oss.str().c_str());
    }

    ret = cudaDeviceSynchronize();
    if (ret)
    {
        std::ostringstream oss;
        oss << "CUDA error : \"" << ret << "\" when draw.";
        throw std::exception(oss.str().c_str());
    }
}
#endif

void Renderer::CommitTextures()
{
    std::vector<float> pixel_buffer;
    std::vector<uint64_t> offsets;
    uint64_t num_pixel = 0;
    for (const Texture::Info &info : list_texture_info_)
    {
        if (info.type == Texture::Type::kBitmap1 ||
            info.type == Texture::Type::kBitmap3 ||
            info.type == Texture::Type::kBitmap4)
        {
            pixel_buffer.insert(pixel_buffer.end(), info.bitmap.data.begin(),
                                info.bitmap.data.end());
            offsets.push_back(num_pixel);
            num_pixel += info.bitmap.data.size();
        }
    }
    try
    {
        DeleteArray(backend_type_, pixels_);
        pixels_ = MallocArray(backend_type_, pixel_buffer);

        DeleteArray(backend_type_, textures_);
        const uint64_t num_texture = list_texture_info_.size();
        textures_ = MallocArray<Texture>(backend_type_, num_texture);
        uint64_t offset_data = 0;
        for (uint64_t i = 0, j = 0; i < num_texture; ++i)
        {
            Texture::Data data;
            data.type = list_texture_info_[i].type;
            switch (data.type)
            {
            case Texture::Type::kConstant:
                data.constant = list_texture_info_[i].constant;
                break;
            case Texture::Type::kCheckerboard:
                data.checkerboard = list_texture_info_[i].checkerboard;
                break;
            case Texture::Type::kBitmap1:
            case Texture::Type::kBitmap3:
            case Texture::Type::kBitmap4:
                data.bitmap.width = list_texture_info_[i].bitmap.width;
                data.bitmap.height = list_texture_info_[i].bitmap.height;
                data.bitmap.to_uv = list_texture_info_[i].bitmap.to_uv;
                data.bitmap.data = pixels_;
                offset_data = offsets[j++];
                break;
            default:
                throw std::exception("unknow texture type.");
                break;
            }
            textures_[i] = Texture(i, data, offset_data);
        }
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit textures to renderer.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::CommitBsdfs()
{
    try
    {
        DeleteArray(backend_type_, bsdfs_);
        const uint64_t num_bsdf = list_bsdf_info_.size();
        bsdfs_ = MallocArray<BSDF>(backend_type_, num_bsdf);
        for (uint64_t i = 0; i < num_bsdf; ++i)
        {
            const BSDF::Info &info = list_bsdf_info_[i];
            CheckTexture(info.id_opacity, true);
            CheckTexture(info.id_bump_map, true);
            switch (info.type)
            {
            case BSDF::Type::kAreaLight:
                CheckTexture(info.area_light.id_radiance, false);
                break;
            case BSDF::Type::kDiffuse:
                CheckTexture(info.diffuse.id_diffuse_reflectance, false);
                break;
            case BSDF::Type::kRoughDiffuse:
                CheckTexture(info.rough_diffuse.id_diffuse_reflectance, false);
                CheckTexture(info.rough_diffuse.id_roughness, false);
                break;
            case BSDF::Type::kConductor:
                CheckTexture(info.conductor.id_roughness_u, false);
                CheckTexture(info.conductor.id_roughness_v, false);
                CheckTexture(info.conductor.id_specular_reflectance, false);
                break;
            case BSDF::Type::kThinDielectric:
            case BSDF::Type::kDielectric:
                CheckTexture(info.dielectric.id_roughness_u, false);
                CheckTexture(info.dielectric.id_roughness_v, false);
                CheckTexture(info.dielectric.id_specular_reflectance, false);
                CheckTexture(info.dielectric.id_specular_transmittance, false);
                break;
            case BSDF::Type::kPlastic:
                CheckTexture(info.plastic.id_roughness, false);
                CheckTexture(info.plastic.id_diffuse_reflectance, false);
                CheckTexture(info.plastic.id_specular_reflectance, false);
                break;
            default:
                throw std::exception("unknow BSDF type.");
                break;
            }
            bsdfs_[i] = BSDF(i, info, textures_);
        }
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit BSDFs to renderer." << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::CommitEmitters()
{
    try
    {
        id_sun_ = kInvalidId;
        id_envmap_ = kInvalidId;
        DeleteArray(backend_type_, emitters_);
        const uint64_t num_emitter = list_emitter_info_.size();
        emitters_ = MallocArray<Emitter>(backend_type_, num_emitter);
        for (uint64_t i = 0; i < num_emitter; ++i)
        {
            const Emitter::Info &info = list_emitter_info_[i];
            switch (info.type)
            {
            case Emitter::Type::kPoint:
            case Emitter::Type::kSpot:
            case Emitter::Type::kDirectional:
                break;
            case Emitter::Type::kSun:
                CheckTexture(info.sun.id_texture, false);
                id_sun_ = i;
                break;
            case Emitter::Type::kEnvMap:
                CheckTexture(info.envmap.id_radiance, false);
            case Emitter::Type::kConstant:
                id_envmap_ = i;
                break;
            default:
                throw std::exception("unknow emitter type.");
                break;
            }
            emitters_[i] = Emitter(i, info, tlas_, textures_);

            if (info.type == Emitter::Type::kEnvMap)
            {
                Texture::Info info_radiance =
                    list_texture_info_[info.envmap.id_radiance];
                if (info_radiance.type != Texture::Type::kBitmap1 &&
                    info_radiance.type != Texture::Type::kBitmap3 &&
                    info_radiance.type != Texture::Type::kBitmap4)
                {
                    std::ostringstream oss;
                    oss << "radiance texture '" << info.envmap.id_radiance
                        << "' for emitter '" << i << "' is not a bitmap.";
                    throw std::exception(oss.str().c_str());
                }

                std::vector<float> cdf_cols, cdf_rows, weight_rows;
                float normalization;

                Emitter::CreateEnvMapCdfPdf(
                    info_radiance.bitmap.width, info_radiance.bitmap.height,
                    textures_[info.envmap.id_radiance], &cdf_cols, &cdf_rows,
                    &weight_rows, &normalization);

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

                emitters_[i].InitEnvMap(info_radiance.bitmap.width,
                                        info_radiance.bitmap.height,
                                        normalization, data_env_map_);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit BSDFs to renderer." << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::CommitIntegrator()
{
    try
    {
        DeleteElement(backend_type_, integrator_);
        integrator_ = MallocElement<Integrator>(backend_type_);

        Integrator::Data data_integrator;

        data_integrator.pdf_rr = info_integrator_.pdf_rr;
        data_integrator.depth_rr = info_integrator_.depth_rr;
        data_integrator.depth_max = info_integrator_.depth_max;

        data_integrator.num_area_light = num_area_light_;
        data_integrator.num_emitter =
            static_cast<uint32_t>(list_emitter_info_.size());
        data_integrator.id_sun = id_sun_;
        data_integrator.id_envmap = id_envmap_;

        data_integrator.bsdfs = bsdfs_;

        data_integrator.instances = instances_;
        data_integrator.list_pdf_area_instance = list_pdf_area_instance_;

        data_integrator.emitters = emitters_;
        data_integrator.map_id_area_light_instance = map_area_light_instance_;
        data_integrator.map_id_instance_area_light = map_instance_area_light_;
        data_integrator.cdf_area_light = cdf_area_light_;

        data_integrator.tlas = tlas_;
        data_integrator.map_instance_bsdf = map_instance_bsdf_;

        *integrator_ = Integrator(data_integrator);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when set integrator to renderer.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::CheckTexture(const uint32_t id, const bool allow_invalid)
{
    if (id == kInvalidId && allow_invalid)
        return;

    if (id >= list_texture_info_.size())
    {
        std::ostringstream oss;
        oss << "cannot find texture (id " << id << ").";
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::CheckBsdf(const uint32_t id, const bool allow_invalid)
{
    if (id == kInvalidId && allow_invalid)
        return;

    if (id >= list_bsdf_info_.size())
    {
        std::ostringstream oss;
        oss << "cannot find BSDF (id " << id << ").";
        throw std::exception(oss.str().c_str());
    }
}

} // namespace csrt
