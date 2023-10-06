#include "ndf.hpp"

#include "../core/ray.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../math/sample.hpp"
#include "../textures/texture.hpp"

NAMESPACE_BEGIN(raytracer)

static constexpr double kStep = 1.0 / kLutResolution;
static constexpr int kSampleCount = 1024;
static constexpr double kSampleCountInv = 1.0 / kSampleCount;

Ndf::Ndf(NdfType type, Texture *alpha_u, Texture *alpha_v)
    : type_(type),
      alpha_u_(alpha_u),
      alpha_v_(alpha_v),
      compensate_(false)
{
}

bool Ndf::UseTextureMapping() const
{
    return !alpha_u_->IsConstant() || !alpha_v_->IsConstant();
}

double Ndf::albdo(double cos_theta) const
{
    double offset = cos_theta * kLutResolution;
    auto offset_int = static_cast<int>(offset);
    if (offset_int >= kLutResolution - 1)
    {
        return albedo_lut_.back();
    }
    else
    {
        return Lerp(offset - offset_int, albedo_lut_[offset_int], albedo_lut_[offset_int + 1]);
    }
}

std::pair<double, double> Ndf::roughness(const dvec2 &texcoord) const
{
    double alpha_u = alpha_u_->color(texcoord).x,
           alpha_v = alpha_v_->color(texcoord).x;
    return {alpha_u, alpha_v};
}

///\brief 预计算光线出射方向与法线方向夹角的余弦从0到1的一系列反照率和平均反照率
void Ndf::ComputeAlbedoTable()
{
    if (UseTextureMapping())
    {
        return;
    }

    albedo_lut_.fill(0);
    dvec3 normal = {0, 0, 1};

    auto [alpha_u, alpha_v] = roughness(dvec2(0));
    if (alpha_u < 0.01 || alpha_v < 0.01)
    {
        return;
    }

    dvec3 h = {0, 1, 0};
    double pdf = 0.0;
    //预计算光线出射方向与法线方向夹角的余弦从0到1的一系列反照率
    for (int j = 0; j < kLutResolution; ++j)
    {
        double cos_theta_o = kStep * (j + 0.5);
        dvec3 wo = {std::sqrt(1.0 - Sqr(cos_theta_o)), 0, cos_theta_o};
        for (int i = 0; i < kSampleCount; ++i)
        {
            Sample(normal, alpha_u, alpha_v, Hammersley(i + 1, kSampleCount + 1), &h, &pdf);
            dvec3 wi = -Reflect(-wo, h);
            double G = SmithG1(-wi, h, normal, alpha_u, alpha_v) *
                       SmithG1(wo, h, normal, alpha_u, alpha_v),
                   cos_m_o = std::max(glm::dot(wo, h), 0.0),
                   cos_m_n = std::max(glm::dot(normal, h), 0.0);
            //重要性采样的微表面模型BSDF，并且菲涅尔项置为1（或0）
            albedo_lut_[j] += (cos_m_o * G / (cos_theta_o * cos_m_n));
        }
        albedo_lut_[j] = std::max(kSampleCountInv * albedo_lut_[j], 0.0);
        albedo_lut_[j] = std::min(albedo_lut_[j], 1.0);
    }

    albedo_avg_ = 0.0;
    //积分，计算平均反照率
    for (int j = 0; j < kLutResolution; ++j)
    {
        double avg_tmp = 0,
               cos_theta_o = kStep * (j + 0.5);
        dvec3 wo = {std::sqrt(1.0 - Sqr(cos_theta_o)), 0, cos_theta_o};
        for (int i = 0; i < kSampleCount; ++i)
        {
            Sample(normal, alpha_u, alpha_v, Hammersley(i + 1, kSampleCount + 1), &h, &pdf);
            dvec3 wi = -Reflect(-wo, h);
            double cos_theta_i = std::max(glm::dot(-wi, normal), 0.0);
            avg_tmp += (albedo_lut_[j] * cos_theta_i);
        }
        albedo_avg_ += (avg_tmp * 2.0 * kSampleCountInv);
    }
    albedo_avg_ *= kStep;
    if (albedo_avg_ > 1.0 - kEpsilonCompare || albedo_avg_ < kEpsilonCompare)
    {
        return;
    }

    compensate_ = true;
}

NAMESPACE_END(raytracer)