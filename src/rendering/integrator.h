#pragma once

#include <memory>

#include "../material/emitters/envmap.h"
#include "../utils/accelerator/bvh_accel.h"
#include "../modeling/scene.h"

NAMESPACE_BEGIN(simple_renderer)

//全局光照模型基类
class Integrator
{
public:
    /**
     * \brief 全局光照模型
     * \param type 全局光照模型类型
     * \param scene 待着色的场景
     * \param max_depth 递归地追踪光线的最大深度
     * \param rr_depth 最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
     */
    Integrator(int max_depth, int rr_depth)
        : max_depth_(max_depth), rr_depth_(rr_depth), pdf_rr_(0.95) {}

    virtual ~Integrator() {}

    void SetScene(Scene *scene)
    {
        envmap_ = scene->envmap();
        emitters_.clear();
        for (auto shape : scene->shapes())
        {
            if (shape->HasEmission())
                emitters_.push_back(shape);
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

protected:
    int max_depth_;                 //递归地追踪光线的最大深度
    int rr_depth_;                  //最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
    Float pdf_rr_;                  //递归地追踪光线俄罗斯轮盘赌的概率
    Envmap *envmap_;                //用于绘制的天空盒
    std::unique_ptr<BvhAccel> bvh_; //层次包围盒
    std::vector<Shape *> emitters_; //包含的发光物体

    /**
     * \brief 按面积直接采样发光物体上一点
     * \param its 采样到的发光物体上一点（输入/输出参数）
     * \return 是否采样成功
     */
    bool SampleEmitterDirectIts(Intersection &its) const
    {
        if (this->emitters_.empty())
            return false;
        auto index = static_cast<int>(UniformFloat2() * this->emitters_.size());
        its = this->emitters_[index]->SampleP();
        return true;
    }

    /**
     * \brief 按面积直接采样发光物体上一点，累计多重重要性采样下直接来自光源的辐射亮度（光亮度）
     * \param its 采样时当前所在的光线与物体表面交点
     * \param wo 采样时当前所在的光线与物体表面交点处，光线的出射方向
     * \param value 直接来自光源的辐射亮度（光亮度） （输入/输出参数）
     */
    bool EmitterDirectArea(const Intersection &its, const Vector3 &wo, Spectrum &value, const Intersection *its_emitter_ptr = nullptr) const
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

        value += weight_direct * its_pre.radiance() * bsdf * cos_theta / pdf_direct;
        return true;
    }

    /**
     * \brief 计算光线从某个发光物体向给定方向射出的概率（相对于光线出射后与物体表面交点处的立体角）
     * \param its_pre 光线从发光物体射出的起点
     * \param wi 光线出射方向
     * \param distance_sqr 光线出射后与物体表面的交点和光源上光线出射点之间距离的平方 （可选参数）
     * \return 光线从某个发光物体向给定方向射出的概率（相对于光线出射后与物体表面交点处的立体角）
     */
    Float PdfEmitterDirect(const Intersection &its_pre, const Vector3 &wi, Float *distance_sqr = nullptr) const
    {
        auto cos_theta_prime = glm::dot(wi, its_pre.normal());
        if (cos_theta_prime < 0)
            return 0;
        auto pdf_area = its_pre.pdf_area() / emitters_.size();
        auto distance_sqr_ = distance_sqr == nullptr ? std::pow(its_pre.distance(), 2) : *distance_sqr;
        return pdf_area * distance_sqr_ / cos_theta_prime;
    }

    /**
     * \brief 判断场景中某两个物体表面点之间是否被遮挡
     * \param its1 待判断是否被遮挡的场景中某个物体表面点
     * \param its2 待判断是否被遮挡的场景中另一个物体表面点
     * \param distance_sqr 两个物体表面点之间距离的平方 （输入/输出参数）
     * \return 两点之间是否被遮挡的结果
     */
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
};

NAMESPACE_END(simple_renderer)