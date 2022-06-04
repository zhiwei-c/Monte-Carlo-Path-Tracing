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
        auto L = Spectrum(0),                 //着色结果
            global_attenuation = Spectrum(1); //光能因被物体吸收而衰减的系数
        size_t depth = 1;                     //光线溯源深度
        Vector3 wo = -look_dir;               //当前出射光线方向
        auto its = Intersection(eye_pos);     //当前散射点
        while (depth < static_cast<size_t>(max_depth_) && (depth <= rr_depth_ || UniformFloat() < pdf_rr_))
        { //迭代地溯源光线
            if (!its.HarshLobe())
            { //按发光物体表面积采样来自面光源的直接光照，生成 shadow rays
                auto L_direct = Spectrum(0);
                EmitterDirectArea(its, wo, L_direct);
                L += L_direct * global_attenuation;
            }

            SamplingRecord b_rec = its.Sample(wo);
            if (b_rec.type == ScatteringType::kNone)
            { //抽样次生光线失败，结束迭代
                break;
            }
            else
            {
                global_attenuation *= b_rec.attenuation / b_rec.pdf;
            }

            its = Intersection();
            if (!this->bvh_->Intersect(Ray(b_rec.pos, -b_rec.wi), its))
            { //按 BSDF 采样来自环境的直接光照
                if (envmap_ != nullptr)
                    L += envmap_->radiance(-b_rec.wi) * global_attenuation;
                break;
            }
            else if (its.absorb())
            { //光线与单面材质的物体交于物体背面而被吸收
                break;
            }
            else if (its.HasEmission())
            { //按 BSDF 采样来自面光源的直接光照
                if (depth == 1 || its.HarshLobe())
                    L += its.radiance() * global_attenuation;
                else
                {
                    Float pdf_direct = PdfEmitterDirect(its, b_rec.wi),
                          weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
                    L += weight_bsdf * its.radiance() * global_attenuation;
                }
                break;
            }

            if (depth > rr_depth_)
            { //处理俄罗斯轮盘赌算法
                global_attenuation /= pdf_rr_;
            }
            wo = b_rec.wi;
            depth++;
        }
        return L;
    }
};
NAMESPACE_END(raytracer)