#pragma once

#include <memory>

#include "../modeling/envmap.h"
#include "../utils/accelerator/bvh_accel.h"
#include "../modeling/scene.h"

NAMESPACE_BEGIN(simple_renderer)

//全局光照模型类型
enum class IntegratorType
{
    kPath, //路径跟踪
    kBdpt, //双向路径跟踪
};

struct DirectSampling
{
    bool valid;
    Intersection its;
    Vector3 wi;
    Spectrum bsdf;
    Float pdf;

    DirectSampling() : valid(false), its(Intersection()), pdf(0), wi(Vector3(0)), bsdf(Spectrum(0)) {}

    DirectSampling(const Intersection &its)
        : valid(true), its(its), pdf(0), wi(Vector3(0)), bsdf(Spectrum(0)) {}

    DirectSampling(const Intersection &its, const Vector3 &wi, const Spectrum &bsdf, const Float pdf)
        : valid(true), its(its), pdf(pdf), wi(wi), bsdf(bsdf) {}
};

//全局光照模型基类
class Integrator
{
public:
    /**
     * \brief 全局光照模型
     * \param type 全局光照模型类型
     * \param max_depth 递归地追踪光线的最大深度
     */
    Integrator(IntegratorType type, int max_depth) : type_(type), max_depth_(max_depth), pdf_rr_(0.95) {}

    /**
     * \brief 设置待着色的场景
     * \param scene 待着色的场景
     */
    void SetScene(Scene *scene)
    {
        envmap_ = scene->envmap();
        emitters_.clear();
        emit_area_ = 0;
        for (auto &shape : scene->shapes())
        {
            if (shape->HasEmission())
            {
                emitters_.push_back(shape);
                emit_area_ += shape->area();
            }
        }
        if (!scene->shapes().empty())
            bvh_ = std::make_unique<BvhAccel>(scene->shapes());
    }

    /**
     * \brief 着色
     * \param eye_pos 观察点的坐标
     * \param look_dir 观察方向
     * \return 观察点来源于给定观察方向的辐射亮度
     */
    virtual Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const = 0;

    ///\brief 全局光照模型类型
    IntegratorType type() const { return type_; }

protected:
    std::unique_ptr<BvhAccel> bvh_; //层次包围盒
    std::vector<Shape *> emitters_; //包含的发光物体
    Float emit_area_;               //发光物体的总表面积
    Envmap *envmap_;                //用于绘制的天空盒
    int max_depth_;                 //递归地追踪光线的最大深度
    Float pdf_rr_;                  //递归地追踪光线俄罗斯轮盘赌的概率

    bool SampleEmitterDirectIts(Intersection &its) const
    {
        if (this->emitters_.empty())
            return false;
        auto index = static_cast<int>(UniformFloat2() * this->emitters_.size());
        std::tie(its, std::ignore) = this->emitters_[index]->SampleP();
        return true;
    }

    void EmitterDirectArea(const Intersection &its, const Vector3 &wo, Spectrum &value) const
    {
        if (this->emitters_.empty())
            return;

        auto index = static_cast<int>(UniformFloat2() * this->emitters_.size());
        auto [its_pre, pdf_area] = this->emitters_[index]->SampleP();
        pdf_area /= this->emitters_.size();

        Float distance_sqr = 0;
        if (!Visible(its_pre, its, &distance_sqr))
            return;

        auto wi = glm::normalize(its.pos() - its_pre.pos());
        auto cos_theta_prime = glm::dot(wi, its_pre.normal());
        if (cos_theta_prime < 0)
            return;

        auto pdf_direct = pdf_area * distance_sqr / cos_theta_prime;

        auto bsdf = its.Eval(wi, wo);
        if (bsdf.r + bsdf.g + bsdf.b < kEpsilon)
            return;

        auto pdf_bsdf = its.Pdf(wi, wo);
        auto weight_direct = MisWeight(pdf_direct, pdf_bsdf);
        auto cos_theta = std::abs(glm::dot(wi, its.normal()));

        value += weight_direct * its_pre.radiance() * bsdf * cos_theta / pdf_direct;
    }

    Float PdfEmitterDirect(const Intersection &its_pre, const Vector3 &wi, Float *distance_sqr = nullptr) const
    {
        auto cos_theta_prime = glm::dot(wi, its_pre.normal());
        if (cos_theta_prime < 0)
            return 0;

        auto pdf_area = 1 / (its_pre.shape_area() * emitters_.size());
        auto distance_sqr_ = distance_sqr == nullptr ? std::pow(its_pre.distance(), 2) : *distance_sqr;
        return pdf_area * distance_sqr_ / cos_theta_prime;
    }

    bool Visible(const Intersection &its1, const Intersection &its2, Float *distance_sqr = nullptr) const
    {
        auto d_vec = its2.pos() - its1.pos();
        auto ray = Ray(its1.pos(), d_vec);
        auto its = this->bvh_->Intersect(ray);
        if (its.distance() + kEpsilonDistance < glm::length(d_vec))
            return false;
        else
        {
            if (distance_sqr != nullptr)
                *distance_sqr = std::pow(its.distance(), 2);
            return true;
        }
    }

private:
    IntegratorType type_; //全局光照模型类型
};

NAMESPACE_END(simple_renderer)