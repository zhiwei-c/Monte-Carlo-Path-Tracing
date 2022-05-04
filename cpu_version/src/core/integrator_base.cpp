#include "integrator_base.h"

NAMESPACE_BEGIN(simple_renderer)

///\brief 初始化全局光照模型
void Integrator::InitIntegrator(std::vector<Shape *> &shapes, Envmap *envmap)
{
    envmap_ = envmap;
    emitters_.clear();
    for (auto shape : shapes)
    {
        if (shape->HasEmission())
            emitters_.push_back(shape);
    }

    if (!shapes.empty())
        bvh_ = std::make_unique<BvhAccel>(shapes);
}

///\brief 按面积直接采样发光物体上一点
bool Integrator::SampleEmitterDirectIts(Intersection &its) const
{
    if (this->emitters_.empty())
        return false;
    auto index = static_cast<int>(UniformFloat2() * this->emitters_.size());
    its = this->emitters_[index]->SampleP();
    return true;
}

///\brief 按面积直接采样发光物体上一点，累计多重重要性采样下直接来自光源的辐射亮度
bool Integrator::EmitterDirectArea(const Intersection &its, const Vector3 &wo, Spectrum &value, const Spectrum *attenuation, const Intersection *its_emitter_ptr) const
{
    if (this->emitters_.empty())
        return false;

    auto its_pre = Intersection();
    if (its_emitter_ptr)
        its_pre = *its_emitter_ptr;
    else
    {
        auto index = static_cast<int>(UniformFloat2() * this->emitters_.size());
        its_pre = this->emitters_[index]->SampleP();
    }

    Float pdf_area = its_pre.pdf_area() / this->emitters_.size();

    Float distance_sqr = 0;
    if (!Visible(its_pre, its, &distance_sqr))
        return false;

    auto wi = glm::normalize(its.pos() - its_pre.pos());
    auto cos_theta_prime = glm::dot(wi, its_pre.normal());
    if (cos_theta_prime < 0)
        return false;

    if (Perpendicular(-wi, its.normal()))
        return false;

    auto pdf_direct = pdf_area * distance_sqr / cos_theta_prime;

    auto bsdf = its.Eval(wi, wo);
    if (bsdf.r + bsdf.g + bsdf.b < kEpsilon)
        return false;

    auto pdf_bsdf = its.Pdf(wi, wo);
    auto weight_direct = MisWeight(pdf_direct, pdf_bsdf);
    auto cos_theta = std::abs(glm::dot(wi, its.normal()));

    if (attenuation)
        value += *attenuation * weight_direct * its_pre.radiance() * bsdf * cos_theta / pdf_direct;
    else
        value += weight_direct * its_pre.radiance() * bsdf * cos_theta / pdf_direct;

    return true;
}

///\brief 计算光线从某个发光物体向给定方向射出的概率（相对于光线出射后与物体表面交点处的立体角）
Float Integrator::PdfEmitterDirect(const Intersection &its_pre, const Vector3 &wi, Float *distance_sqr) const
{
    auto cos_theta_prime = glm::dot(wi, its_pre.normal());
    if (cos_theta_prime < 0)
        return 0;
    auto pdf_area = its_pre.pdf_area() / emitters_.size();
    auto distance_sqr_ = distance_sqr == nullptr ? std::pow(its_pre.distance(), 2) : *distance_sqr;
    return pdf_area * distance_sqr_ / cos_theta_prime;
}

///\brief 判断场景中某两个物体表面点之间是否被遮挡
bool Integrator::Visible(const Intersection &its1, const Intersection &its2, Float *distance_sqr) const
{
    auto d_vec = its2.pos() - its1.pos();
    auto ray = Ray(its1.pos(), d_vec);
    Intersection its;
    this->bvh_->Intersect(ray, its);
    if (its.distance() + kEpsilonDistance < glm::length(d_vec))
        return false;
    else
    {
        if (distance_sqr != nullptr)
            *distance_sqr = std::pow(its.distance(), 2);
        return true;
    }
}

NAMESPACE_END(simple_renderer)