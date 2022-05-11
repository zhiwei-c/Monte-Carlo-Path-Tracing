#pragma once

#include "../core/integrator_base.h"

NAMESPACE_BEGIN(raytracer)

//路径追踪算法类
class PathIntegrator : public Integrator
{
public:
    ///\brief 路径追踪算法类
    ///\param max_depth 溯源光线的最大跟踪深度
    ///\param rr_depth 最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
    PathIntegrator(int max_depth, int rr_depth) : Integrator(max_depth, rr_depth) {}

    ///\brief 着色
    ///\param eye_pos 观察点的位置
    ///\param look_dir 观察方向
    ///\return 观察点来源于给定观察方向的辐射亮度
    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override
    {
        auto its = Intersection();

        //原初光线源于环境
        if (!this->bvh_ || !this->bvh_->Intersect(Ray(eye_pos, look_dir), its))
            return envmap_ ? envmap_->radiance(look_dir) : Spectrum(0);

        ///单面材质物体的背面，只吸收而不反射或折射光线
        if (its.absorb())
            return Spectrum(0);

        //原初光线源于发光物体
        if (its.HasEmission())
            return its.radiance();

        Vector3 wo = -look_dir;
        size_t depth = 1;              //光线溯源深度
        auto L = Spectrum(0),          //着色结果
            attenuation = Spectrum(1); //光能因被物体吸收而衰减的系数
        auto its_pre = Intersection();
        //迭代地溯源光线
        while (depth < max_depth_ && (depth <= rr_depth_ || UniformFloat() < pdf_rr_))
        {
            //按发光物体表面积采样来自面光源的直接光照
            if (!its.HarshLobe())
                EmitterDirectArea(its, wo, L, &attenuation);

            std::unique_ptr<BsdfSampling> b_rec = its.Sample(wo);
            if (!b_rec)
                break;
            Float cos_theta = glm::abs(glm::dot(b_rec->wi, its.normal()));
            its_pre = Intersection();
            if (!this->bvh_->Intersect(Ray(its.pos(), -b_rec->wi), its_pre))
            {
                if (envmap_ != nullptr) //按 BSDF 采样来自环境的直接光照
                    L += attenuation * envmap_->radiance(-b_rec->wi) * b_rec->attenuation * cos_theta / b_rec->pdf;
                break;
            }
            else if (its_pre.absorb()) //光线与单面材质的物体交于物体背面而被吸收
                break;
            else if (its_pre.HasEmission()) //按 BSDF 采样来自面光源的直接光照
            {
                if (its.HarshLobe())
                    L += attenuation * its_pre.radiance() * b_rec->attenuation * cos_theta / b_rec->pdf;
                else
                {
                    Float pdf_direct = PdfEmitterDirect(its_pre, b_rec->wi),
                          weight_bsdf = MisWeight(b_rec->pdf, pdf_direct);
                    L += attenuation * weight_bsdf * its_pre.radiance() * b_rec->attenuation * cos_theta / b_rec->pdf;
                }
                break;
            }
            //按 BSDF 采样间接光照更新衰减系数
            attenuation *= b_rec->attenuation * cos_theta / b_rec->pdf;
            if (depth > rr_depth_)
                attenuation /= pdf_rr_;

            its = its_pre,
            wo = b_rec->wi,
            depth++;
        }
        return L;
    }
};

NAMESPACE_END(raytracer)