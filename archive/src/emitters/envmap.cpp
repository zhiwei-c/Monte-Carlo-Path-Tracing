#include "envmap.hpp"

#include "../accelerators/accelerator.hpp"
#include "../math/coordinate.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../textures/texture.hpp"
#include "../utils/image.hpp"

NAMESPACE_BEGIN(raytracer)

Envmap::Envmap(Texture *background, double scale, dmat4 to_world)
    : Emitter(EmitterType::kEnvmap),
      background_(background),
      scale_(scale),
      to_local_(glm::inverse(to_world)),
      to_world_(to_world),
      height_(1),
      width_(1),
      height_rcp_(1),
      width_rcp_(1)
{
    if (!background_->IsBitmap())
    {
        use_importance_sampling_ = false;
        return;
    }

    width_ = background_->width(),
    height_ = background_->height();
    width_rcp_ = 1.0 / width_,
    height_rcp_ = 1.0 / height_;

    cdf_rows_ = std::vector<double>(height_ + 1),
    weight_rows_ = std::vector<double>(height_),
    cdf_cols_ = std::vector<std::vector<double>>(height_, std::vector<double>(width_ + 1));

    double row_sum = 0.0;
    size_t row_pos = 0;
    cdf_rows_[row_pos++] = 0;
    for (int y = 0; y < height_; ++y)
    {
        double col_sum = 0;
        size_t col_pos = 0;
        cdf_cols_[y][col_pos++] = 0;
        for (int x = 0; x < width_; ++x)
        {
            col_sum += background_->luminance(dvec2{x * width_rcp_, y * height_rcp_});
            cdf_cols_[y][col_pos++] = col_sum;
        }

        double normalization = 1.0 / col_sum;
        for (int x = 1; x < width_; ++x)
        {
            cdf_cols_[y][col_pos - x - 1] *= normalization;
        }
        cdf_cols_[y][col_pos - 1] = 1.0f;

        double weight = std::sin((y + 0.5f) * kPi / height_);
        weight_rows_[y] = weight;
        row_sum += col_sum * weight;
        cdf_rows_[row_pos++] = row_sum;
    }

    double normalization = 1.0 / row_sum;
    for (int y = 1; y < height_; ++y)
    {
        cdf_rows_[row_pos - y - 1] *= normalization;
    }
    cdf_rows_[row_pos - 1] = 1.0;
    if (row_sum == 0)
    {
        use_importance_sampling_ = false;
        return;
    }
    if (!std::isfinite(row_sum))
    {
        std::cerr << "[error] The environment map contains an invalid floating point value (nan/inf).\n";
        exit(1);
    }

    normalization_ = 1.0 / (row_sum * (2 * kPi * width_rcp_) * (kPi * height_rcp_));
    use_importance_sampling_ = true;
}

SamplingRecord Envmap::Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                              Accelerator *accelerator) const
{
    auto rec = SamplingRecord();
    rec.position = its_shape.position();
    rec.wo = wo;
    size_t row = 0, col = 0;
    double phi = 0, theta = 0;
    if (use_importance_sampling_)
    {
        row = SampleCdf(cdf_rows_, height_, sampler->Next1D());
        col = SampleCdf(cdf_cols_[row], width_, sampler->Next1D());
        theta = row * height_rcp_ * kPi;
        phi = 2 * kPi * col * width_rcp_;
        rec.wi = TransfromVec(to_world_, SphericalToCartesian(theta, phi));
    }
    else
    {
        phi = 2 * kPi * sampler->Next1D();
        phi = kPi - phi;
        double cos_theta = 1.0 - 2.0 * sampler->Next1D(),
               sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
        theta = glm::acos(cos_theta);
        rec.wi = dvec3{sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
    }

    if (accelerator != nullptr)
    {
        const Ray ray = {its_shape.position(), -rec.wi};
        auto its_test = Intersection();
        if (accelerator->Intersect(ray, sampler, &its_test))
        {
            return rec;
        }
    }

    const dvec3 color = background_->color(dvec2{phi * 0.5 * kPiRcp, theta * kPiRcp});
    if (use_importance_sampling_)
    {
        rec.pdf = LinearRgbToLuminance(color) * weight_rows_[row] * normalization_ / std::max(std::abs(std::sin(theta)), 1e-4);
    }
    else
    {
        rec.pdf = 0.25 * kPiRcp;
    }

    if (rec.pdf == 0.0)
    {
        return rec;
    }
    rec.type = ScatteringType::kScattering;
    rec.radiance = scale_ * color;
    return rec;
}
double Envmap::Pdf(const dvec3 &look_dir) const
{
    if (use_importance_sampling_)
    {
        const dvec3 dir = TransfromVec(to_local_, look_dir);
        double theta, phi;
        CartesianToSpherical(look_dir, &theta, &phi);
        phi = kPi - phi;
        const dvec2 texcoord = {phi * 0.5 * kPiRcp, theta * kPiRcp};
        double row = std::min(std::max(texcoord.y * height_, 0.0), height_ - 1.0);
        int row_int = static_cast<int>(row);
        double t = row - row_int;
        if (t == 0.0)
        {
            return background_->luminance(texcoord) * weight_rows_[row_int] * normalization_ /
                   std::max(std::abs(std::sin(theta)), 1e-4);
        }
        else
        {
            return background_->luminance(texcoord) * Lerp(t, weight_rows_[row_int], weight_rows_[row_int + 1]) *
                   normalization_ / std::max(std::abs(std::sin(theta)), 1e-4);
        }
    }
    else
    {
        return 0.25 * kPiRcp;
    }
}

dvec3 Envmap::radiance(const dvec3 &position, const dvec3 &wi) const
{
    if (background_->IsConstant())
    {
        return scale_ * background_->color(dvec2{0, 0});
    }

    const dvec3 look_dir = TransfromVec(to_local_, -wi);
    double theta, phi;
    CartesianToSpherical(look_dir, &theta, &phi);
    phi = kPi - phi;
    const dvec2 texcoord = {phi * 0.5 * kPiRcp, theta * kPiRcp};
    return scale_ * background_->color(texcoord);
}

NAMESPACE_END(raytracer)