#include "render.cuh"

#include <algorithm>
#include <array>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>

namespace
{

using namespace rt;

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
    const uint32_t pixel_index = j * camera->width() + i;
    uint32_t seed = Tea<4>(pixel_index, 0);
    Vec3 color;
    for (uint32_t s = 0; s < camera->spp(); ++s)
    {
        const float u = s * camera->spp_inv(),
                    v = GetVanDerCorputSequence<2>(s + 1),
                    x = 2.0f * (i + u) / camera->width() - 1.0f,
                    y = 1.0f - 2.0f * (j + v) / camera->height();
        const Vec3 look_dir = Normalize(
            camera->front() + x * camera->view_dx() + y * camera->view_dy());
        color += integrator->Shade(camera->eye(), look_dir, &seed);
    }
    color *= camera->spp_inv();
    for (int channel = 0; channel < 3; ++channel)
    {
        color[channel] = fminf(color[channel], 1.0f);
        frame[pixel_index * 3 + channel] =
            (color[channel] <= 0.0031308f)
                ? (12.92f * color[channel])
                : (1.055f * powf(color[channel], 1.0f / 2.4f) - 0.055f);
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
#endif

void DispathRaysCpu(Camera *camera, Integrator *integrator, float *frame)
{
    Timer timer;
    uint64_t count_pacth = 0;
    const uint64_t size_patch = m_patches.size();
    const double size_patch_rcp = 1.0 / size_patch;

#if defined(DEBUG) || defined(_DEBUG)
    DrawPixel(939, 579, camera, integrator, frame);
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

namespace rt
{

Renderer::Renderer(const BackendType backend_type)
    : backend_type_(backend_type), list_texture_(nullptr), list_pixel_(nullptr),
      list_bsdf_(nullptr), map_instance_bsdf_(nullptr), num_area_light_(0),
      map_id_area_light_instance_(nullptr),
      map_id_instance_area_light_(nullptr), cdf_area_light_(nullptr),
      instances_(nullptr), list_pdf_area_instance_(nullptr), tlas_(nullptr),
      integrator_(nullptr), camera_(nullptr)
{
}

Renderer::~Renderer()
{
    // release texture
    DeleteArray(backend_type_, list_texture_);
    DeleteArray(backend_type_, list_pixel_);

    // release BSDF
    DeleteArray(backend_type_, list_bsdf_);
    DeleteArray(backend_type_, map_instance_bsdf_);

    DeleteArray(backend_type_, map_id_area_light_instance_);
    DeleteArray(backend_type_, map_id_instance_area_light_);
    DeleteArray(backend_type_, cdf_area_light_);
    DeleteElement(backend_type_, camera_);
    DeleteElement(backend_type_, integrator_);
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

void Renderer::AddBsdf(const Bsdf::Info &info)
{
    switch (info.type)
    {
    case Bsdf::Type::kAreaLight:
    case Bsdf::Type::kDiffuse:
    case Bsdf::Type::kRoughDiffuse:
    case Bsdf::Type::kConductor:
    case Bsdf::Type::kDielectric:
    case Bsdf::Type::kThinDielectric:
    case Bsdf::Type::kPlastic:
        break;
    default:
        throw std::exception("unknow BSDF type.");
        break;
    }

    list_bsdf_info_.push_back(info);
}

void Renderer::AddSceneInfo(const std::vector<uint32_t> &map_id_instance_bsdf,
                            Instance *instances, float *list_pdf_area_instance,
                            TLAS *tlas)
{
    try
    {
        num_instance_ = map_id_instance_bsdf.size();
        tlas_ = tlas;
        instances_ = instances;
        list_pdf_area_instance_ = list_pdf_area_instance;

        for (uint32_t i = 0; i < num_instance_; ++i)
            CheckBsdf(map_id_instance_bsdf[i], true);
        DeleteArray(backend_type_, map_instance_bsdf_);
        map_instance_bsdf_ =
            MallocArray<uint32_t>(backend_type_, map_id_instance_bsdf);
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when add scene info to renderer.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void rt::Renderer::SetAreaLightInfo(
    const std::vector<uint32_t> map_id_area_light_instance,
    const std::vector<float> list_area_light_weight)
{
    try
    {
        DeleteArray(backend_type_, map_id_area_light_instance_);
        map_id_area_light_instance_ =
            MallocArray<uint32_t>(backend_type_, map_id_area_light_instance);

        DeleteArray(backend_type_, map_id_instance_area_light_);
        map_id_instance_area_light_ =
            MallocArray<uint32_t>(backend_type_, num_instance_);
        for (uint32_t i = 0; i < num_instance_; ++i)
            map_id_instance_area_light_[i] = kInvalidId;

        DeleteArray(backend_type_, cdf_area_light_);
        num_area_light_ = static_cast<uint32_t>(list_area_light_weight.size());
        cdf_area_light_ =
            MallocArray<float>(backend_type_, num_area_light_ + 1);
        cdf_area_light_[0] = 0;
        for (uint32_t i = 0; i < num_area_light_; ++i)
        {
            cdf_area_light_[i + 1] =
                list_area_light_weight[i] + cdf_area_light_[i];
            map_id_instance_area_light_[map_id_area_light_instance[i]] = i;
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
        CommitIntegrator();

        if (backend_type_ == BackendType::kCpu)
        {
            GeneratePatchInfo(camera_->width(), camera_->height());
        }
        else
        {
            m_num_blocks = {
                static_cast<unsigned int>(camera_->width() / 8 + 1),
                static_cast<unsigned int>(camera_->height() / 8 + 1), 1};
        }
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

        if (backend_type_ == BackendType::kCpu)
        {
            DispathRaysCpu(camera_, integrator_, frame);
        }
        else
        {
            Timer timer;
            MallocArray<float>(backend_type_,
                               camera_->height() * camera_->width() * 3);
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
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when draw.\n\t" << e.what();
        throw std::exception(oss.str().c_str());
    }
}

void Renderer::CommitTextures()
{
    std::vector<float> pixel_buffer;
    std::vector<uint64_t> offsets;
    uint64_t offset = 0;
    for (const Texture::Info &info : list_texture_info_)
    {
        if (info.type == Texture::Type::kBitmap1 ||
            info.type == Texture::Type::kBitmap3 ||
            info.type == Texture::Type::kBitmap4)
        {
            pixel_buffer.insert(pixel_buffer.end(), info.bitmap.data.begin(),
                                info.bitmap.data.end());
            offsets.push_back(offset);
            offset += info.bitmap.data.size();
        }
    }
    try
    {
        DeleteArray(backend_type_, list_pixel_);
        list_pixel_ = MallocArray(backend_type_, pixel_buffer);

        DeleteArray(backend_type_, list_texture_);
        const uint64_t num_texture = list_texture_info_.size();
        list_texture_ = MallocArray<Texture>(backend_type_, num_texture);
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
                data.bitmap.data = list_pixel_;
                data.bitmap.offset = offsets[j++];
            default:
                throw std::exception("unknow texture type.");
                break;
            }
            list_texture_[i] = Texture(i, data);
        }
    }
    catch (const std::exception &e)
    {
        std::ostringstream oss;
        oss << "error when commit textures to renderer.";
        throw std::exception(oss.str().c_str());
    }
}

void rt::Renderer::CommitBsdfs()
{
    try
    {
        DeleteArray(backend_type_, list_bsdf_);
        const uint64_t num_bsdf = list_bsdf_info_.size();
        list_bsdf_ = MallocArray<Bsdf>(backend_type_, num_bsdf);
        for (uint64_t i = 0; i < num_bsdf; ++i)
        {
            Bsdf::Data data;
            const Bsdf::Info &info = list_bsdf_info_[i];
            data.type = info.type;
            data.twosided = info.twosided;
            CheckTexture(info.id_opacity, true),
                data.id_opacity = info.id_opacity;
            CheckTexture(info.id_bump_map, true),
                data.id_bump_map = info.id_bump_map;
            data.texture_buffer = list_texture_;
            switch (data.type)
            {
            case Bsdf::Type::kAreaLight:
                CheckTexture(info.area_light.id_radiance, false);
                data.area_light.id_radiance = info.area_light.id_radiance;
                break;
            case Bsdf::Type::kDiffuse:
                CheckTexture(info.diffuse.id_diffuse_reflectance, false);
                data.diffuse = info.diffuse;
                break;
            case Bsdf::Type::kRoughDiffuse:
                CheckTexture(info.rough_diffuse.id_diffuse_reflectance, false);
                CheckTexture(info.rough_diffuse.id_roughness, false);
                data.rough_diffuse = info.rough_diffuse;
                break;
            case Bsdf::Type::kConductor:
                CheckTexture(info.conductor.id_roughness_u, false);
                CheckTexture(info.conductor.id_roughness_v, false);
                CheckTexture(info.conductor.id_specular_reflectance, false);
                data.conductor = info.conductor;
                break;
            case Bsdf::Type::kThinDielectric:
            case Bsdf::Type::kDielectric:
                data.twosided = true;
                CheckTexture(info.dielectric.id_roughness_u, false);
                CheckTexture(info.dielectric.id_roughness_v, false);
                CheckTexture(info.dielectric.id_specular_reflectance, false);
                CheckTexture(info.dielectric.id_specular_transmittance, false);
                data.dielectric.id_roughness_u = info.dielectric.id_roughness_u;
                data.dielectric.id_roughness_v = info.dielectric.id_roughness_v;
                data.dielectric.id_specular_reflectance =
                    info.dielectric.id_specular_reflectance;
                data.dielectric.id_specular_transmittance =
                    info.dielectric.id_specular_transmittance;
                data.dielectric.eta = info.dielectric.eta;
                data.dielectric.eta_inv = 1.0f / info.dielectric.eta;
                data.dielectric.reflectivity =
                    (Sqr(info.dielectric.eta - 1.0f) /
                     Sqr(info.dielectric.eta + 1.0f));
                break;
            case Bsdf::Type::kPlastic:
                CheckTexture(info.plastic.id_roughness, false);
                CheckTexture(info.plastic.id_diffuse_reflectance, false);
                CheckTexture(info.plastic.id_specular_reflectance, false);
                data.plastic.reflectivity = (Sqr(info.plastic.eta - 1.0f) /
                                             Sqr(info.plastic.eta + 1.0f));
                data.plastic.id_roughness = info.plastic.id_roughness;
                data.plastic.id_diffuse_reflectance =
                    info.plastic.id_diffuse_reflectance;
                data.plastic.id_specular_reflectance =
                    info.plastic.id_specular_reflectance;
                data.plastic.F_avg = Bsdf::AverageFresnel(info.plastic.eta);
                break;
            default:
                throw std::exception("unknow BSDF type.");
                break;
            }
            list_bsdf_[i] = Bsdf(i, data);
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
        data_integrator.map_id_area_light_instance =
            map_id_area_light_instance_;
        data_integrator.map_id_instance_area_light =
            map_id_instance_area_light_;
        data_integrator.cdf_area_light = cdf_area_light_;
        data_integrator.instances = instances_;
        data_integrator.list_pdf_area_instance = list_pdf_area_instance_;
        data_integrator.tlas = tlas_;
        data_integrator.bsdfs = list_bsdf_;
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

} // namespace rt
