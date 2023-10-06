#include "integrator.hpp"

#include <algorithm>
#include <iostream>
#include <thread>
#include <mutex>

#include "../accelerators/accelerator.hpp"
#include "../emitters/emitter.hpp"
#include "../shapes/shape.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../utils/timer.hpp"

NAMESPACE_BEGIN(raytracer)

static std::mutex patch_mutex;
static dvec3 view_dx, view_dy;
static int spp_x, spp_y, spp_stratified, spp_r;
static double spp_rcp, len_dx, len_dy, width_rcp, height_rcp;

void ShadePatch(const std::vector<std::vector<std::array<int, 3>>> &patches, const Integrator *integrator,
                const Camera &camera, std::vector<float> &frame_buffer, int &pacth_count, Timer &timer);
void Test(const Integrator *integrator, const Camera &camera);

Integrator::Integrator(IntegratorType type, int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
                       const std::vector<Emitter *> &emitters, size_t shape_num)
    : type_(type),
      max_depth_(static_cast<size_t>(max_depth)),
      rr_depth_(static_cast<size_t>(rr_depth)),
      hide_emitters_(hide_emitters),
      accelerator_(accelerator)
{
    envmap_ = nullptr;
    size_t area_light_num = 0;
    for (Emitter *emitter : emitters)
    {
        switch (emitter->type())
        {
        case EmitterType::kArea:
            ++area_light_num;
            area_lights_.push_back(emitter);
            break;
        case EmitterType::kPoint:
        case EmitterType::kSpot:
            point_lights_.push_back(emitter);
            break;
        case EmitterType::kDirectional:
        case EmitterType::kSun:
            directional_lights_.push_back(emitter);
            break;
        case EmitterType::kEnvmap:
        case EmitterType::kSky:
            if (envmap_ != nullptr)
            {
                std::cerr << "[error] find multiple Envmap\n";
                exit(1);
            }
            envmap_ = emitter;
            break;
        default:
            std::cerr << "[error] unknow emitter type\n";
            exit(1);
            break;
        }
    }
    area_light_num_rcp_ = 1.0 / area_light_num;
    no_emitter_num_rcp_ = 1.0 / (shape_num - area_light_num);
}

std::vector<float> Integrator::Shade(const Camera &camera) const
{
    view_dx = camera.right * std::tan(glm::radians(0.5 * camera.fov_x)),
    view_dy = camera.up * std::tan(glm::radians(0.5 * camera.fov_y));

    spp_x = static_cast<int>(std::sqrt(camera.spp)),
    spp_y = camera.spp / spp_x,
    spp_stratified = spp_x * spp_y,
    spp_r = camera.spp - spp_stratified;

    spp_rcp = 1.0 / camera.spp,
    len_dx = 1.0 / spp_x,
    len_dy = 1.0 / spp_y,
    width_rcp = 1.0 / camera.width,
    height_rcp = 1.0 / camera.height;

    const int resolution = camera.width * camera.height;
    auto pixels = std::vector<std::array<int, 3>>(resolution);
    for (int x = 0; x < camera.width; ++x)
    {
        for (int y = 0; y < camera.height; ++y)
        {
            pixels[x * camera.height + y] = {x, y, static_cast<int>(GetMortonCode(x, y))};
        }
    }
    std::sort(pixels.begin(), pixels.end(), [](const std::array<int, 3> &a, const std::array<int, 3> &b)
              { return a.at(2) < b.at(2); });

    // const int patch_size = std::max(64 * 64 / camera.spp, 1);
    const int patch_size = 64;
    const int patch_num = static_cast<int>(pixels.size() + patch_size - 1) / patch_size;
    auto patches = std::vector<std::vector<std::array<int, 3>>>(patch_num);
    for (int i = 0; i < patch_num; ++i)
    {
        int begin = i * patch_size,
            end = std::min((i + 1) * patch_size, resolution);
        patches[i] = std::vector<std::array<int, 3>>(pixels.begin() + begin, pixels.begin() + end);
    }

#ifndef NDEBUG
    Test(this, camera);
#endif

    Timer timer;
    int pacth_count = 0;
    std::vector<std::thread> workers;
    auto frame_buffer = std::vector<float>(camera.width * camera.height * 3, 0);
    int thread_num = std::max(static_cast<int>(std::thread::hardware_concurrency()), 1);
    for (int i = 0; i < thread_num; ++i)
    {
        workers.push_back(std::thread{ShadePatch, patches, this, camera, std::ref(frame_buffer), std::ref(pacth_count),
                                      std::ref(timer)});
    }
    for (int i = 0; i < thread_num; ++i)
    {
        workers[i].join();
    }
    timer.PrintTimePassed("rendering");
    return frame_buffer;
}

dvec3 Integrator::SampleOtherEmittersDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const
{
    auto L = dvec3(0);

    if (envmap_ != nullptr)
    {
        SamplingRecord direct_rec = envmap_->Sample(its_shape, wo, sampler, accelerator_);
        if (direct_rec.type != ScatteringType::kNone)
        {
            SamplingRecord bsdf_rec = its_shape.Eval(direct_rec.wi, wo);
            if (bsdf_rec.type != ScatteringType::kNone)
            {
                L += MisWeight(direct_rec.pdf, bsdf_rec.pdf) * direct_rec.radiance *
                     bsdf_rec.attenuation / direct_rec.pdf;
            }
        }
    }

    for (const std::vector<raytracer::Emitter *> &lights : {point_lights_, directional_lights_})
    {
        for (Emitter *const &emitter : lights)
        {
            SamplingRecord direct_rec = emitter->Sample(its_shape, wo, sampler, accelerator_);
            if (direct_rec.type == ScatteringType::kNone)
            {
                continue;
            }

            SamplingRecord bsdf_rec = its_shape.Eval(direct_rec.wi, wo);
            if (bsdf_rec.type == ScatteringType::kNone)
            {
                continue;
            }

            L += direct_rec.radiance * bsdf_rec.attenuation;
        }
    }
    return L;
}

dvec3 Integrator::SampleAreaLightsDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const
{
    if (its_shape.IsHarshLobe() || area_lights_.empty())
    {
        return dvec3(0);
    }

    int index = static_cast<int>(sampler->Next1D() * area_lights_.size());
    if (index == area_lights_.size())
    {
        --index;
    }

    SamplingRecord direct_rec = area_lights_[index]->Sample(its_shape, wo, sampler, accelerator_);
    if (direct_rec.type == ScatteringType::kNone)
    {
        return dvec3(0);
    }

    SamplingRecord bsdf_rec = its_shape.Eval(direct_rec.wi, wo);
    if (bsdf_rec.type == ScatteringType::kNone)
    {
        return dvec3(0);
    }

    double pdf_direct = direct_rec.pdf * area_light_num_rcp_;
    if (pdf_direct <= kEpsilonPdf)
    {
        return dvec3(0);
    }

    return MisWeight(pdf_direct, bsdf_rec.pdf) * direct_rec.radiance * bsdf_rec.attenuation / pdf_direct;
}

double Integrator::PdfAreaLight(const Intersection &its_light, const dvec3 &wi) const
{
    double cos_theta_prime = glm::dot(wi, its_light.normal());
    if (cos_theta_prime < 0)
    {
        return 0;
    }
    double pdf_area = its_light.pdf_area() * area_light_num_rcp_;
    return pdf_area * its_light.distance() * its_light.distance() / cos_theta_prime;
}

double Integrator::PdfEnvmap(const dvec3 &wi) const
{
    return envmap_->Pdf(wi);
}

bool Integrator::Visible(const Intersection &a, const Intersection &b, Sampler *sampler) const
{
    const double distance = glm::length(b.position() - a.position());
    const dvec3 ray_dir = glm::normalize(b.position() - a.position());
    const Ray ray = {a.position(), ray_dir, distance};
    auto its_test = Intersection();
    return !accelerator_->Intersect(ray, sampler, &its_test) ||
           its_test.shape_id() == b.shape_id();
}

void ShadePatch(const std::vector<std::vector<std::array<int, 3>>> &patches, const Integrator *integrator,
                const Camera &camera, std::vector<float> &frame_buffer, int &pacth_count, Timer &timer)
{
#ifndef NDEBUG
    const unsigned int val2 = 0;
#else
    time_t time_now;
    time(&time_now);
    const auto val2 = static_cast<unsigned int>(time_now % std::numeric_limits<unsigned int>::max());
#endif
    int patch_id = 0;
    const double patch_num_rcp = 1.0 / patches.size();
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(patch_mutex);
            if (pacth_count == patches.size())
            {
                break;
            }
            patch_id = pacth_count++;
        }
        for (const std::array<int, 3> &pixel : patches[patch_id])
        {
            const int &x = pixel.at(0),
                      &y = pixel.at(1),
                      pixel_offset = (x + y * camera.width) * 3;
            dvec3 color = dvec3(0),
                  look_dir = dvec3(0);
            unsigned int spp_count = 0;
            for (int i = 0; i < spp_x; ++i)
            {
                for (int j = 0; j < spp_y; ++j)
                {
                    unsigned int seed = Tea(pixel_offset + val2, spp_count++, 16);
                    auto sampler = Sampler(seed);
                    double offset_x = x + (i + sampler.Next1D()) * len_dx,
                           offset_y = y + (j + sampler.Next1D()) * len_dy;
                    offset_x = 2.0 * offset_x * width_rcp - 1.0,
                    offset_y = 1.0 - 2.0 * offset_y * height_rcp;
                    look_dir = glm::normalize(camera.front + offset_x * view_dx + offset_y * view_dy);
                    color += integrator->Shade(camera.eye, look_dir, &sampler);
                }
            }
            for (int i = 0; i < spp_r; ++i)
            {
                unsigned int seed = Tea(pixel_offset + val2, spp_count++, 16);
                auto sampler = Sampler(seed);
                double offset_x = 2.0 * (x + sampler.Next1D()) * width_rcp - 1.0,
                       offset_y = 1.0 - 2.0 * (y + sampler.Next1D()) * height_rcp;
                look_dir = glm::normalize(camera.front + offset_x * view_dx + offset_y * view_dy);
                color += integrator->Shade(camera.eye, look_dir, &sampler);
            }
            color *= spp_rcp;
            frame_buffer[pixel_offset] = static_cast<float>(color.r),
            frame_buffer[pixel_offset + 1] = static_cast<float>(color.g),
            frame_buffer[pixel_offset + 2] = static_cast<float>(color.b);
        }
        {
            std::lock_guard<std::mutex> lock(patch_mutex);
            timer.PrintProgress(pacth_count * patch_num_rcp);
        }
    }
}

void Test(const Integrator *integrator, const Camera &camera)
{
    const int &x = 666,
              &y = 840,
              pixel_offset = (x + y * camera.width) * 3;
    dvec3 color = dvec3(0),
          look_dir = dvec3(0);
    unsigned int spp_count = 0;
    for (int i = 0; i < spp_x; ++i)
    {
        for (int j = 0; j < spp_y; ++j)
        {
            unsigned int seed = Tea(pixel_offset, spp_count++, 16);
            auto sampler = Sampler(seed);
            double offset_x = x + (i + sampler.Next1D()) * len_dx,
                   offset_y = y + (j + sampler.Next1D()) * len_dy;

            offset_x = 2.0 * offset_x * width_rcp - 1.0,
            offset_y = 1.0 - 2.0 * offset_y * height_rcp;
            look_dir = glm::normalize(camera.front + offset_x * view_dx + offset_y * view_dy);
            auto color_local = integrator->Shade(camera.eye, look_dir, &sampler);
            std::cout << "spp " << spp_count << ", seed = " << seed
                      << ",color = (" << color_local.x << ", " << color_local.y << ", " << color_local.z << ")\n";
            color += color_local;
        }
    }
    for (int i = 0; i < spp_r; ++i)
    {
        unsigned int seed = Tea(pixel_offset, spp_count++, 16);
        auto sampler = Sampler(seed);
        double offset_x = 2.0 * (x + sampler.Next1D()) * width_rcp - 1.0,
               offset_y = 1.0 - 2.0 * (y + sampler.Next1D()) * height_rcp;
        look_dir = glm::normalize(camera.front + offset_x * view_dx + offset_y * view_dy);
        auto color_local = integrator->Shade(camera.eye, look_dir, &sampler);
        std::cout << "spp " << spp_count << ", seed = " << seed
                  << ",color = (" << color_local.x << ", " << color_local.y << ", " << color_local.z << ")\n";
        color += color_local;
    }
    color *= spp_rcp;
    std::cout << "color = (" << color.x << ", " << color.y << ", " << color.z << ")\n";
    return;
}

NAMESPACE_END(raytracer)