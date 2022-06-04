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
        size_t depth = 1;                     //光线溯源深度
        auto L = Spectrum(0),                 //着色结果
            global_attenuation = Spectrum(1); //光能因被物体吸收而衰减的系数
        Vector3 wo = -look_dir;
        auto its = Intersection();

        if (!this->bvh_ || !this->bvh_->Intersect(Ray(eye_pos, look_dir), its))
        { //原初光线源于环境
            return envmap_ ? envmap_->radiance(look_dir) : Spectrum(0);
        }
        else if (its.absorb())
        { ///单面材质物体的背面，只吸收而不反射或折射光线
            return Spectrum(0);
        }
        else if (its.HasEmission())
        { //原初光线源于发光物体
            return its.radiance();
        }

        auto its_pre = Intersection();
        while (depth < max_depth_ && (depth <= rr_depth_ || UniformFloat() < pdf_rr_))
        { //迭代地溯源光线
            if (!its.HarshLobe())
            { //按发光物体表面积采样来自面光源的直接光照，生成 shadow rays
                EmitterDirectArea(its, wo, L, &global_attenuation);
            }

            SamplingRecord b_rec = its.Sample(wo);
            if (b_rec.type == ScatteringType::kNone)
            { //抽样次生光线失败，结束迭代
                break;
            }

            auto local_attenuation = b_rec.attenuation / b_rec.pdf;
            its_pre = Intersection();
            if (!this->bvh_->Intersect(Ray(b_rec.pos, -b_rec.wi), its_pre))
            { //按 BSDF 采样来自环境的直接光照
                if (envmap_ != nullptr)
                    L += global_attenuation * envmap_->radiance(-b_rec.wi) * local_attenuation;
                break;
            }
            else if (its_pre.absorb())
            { //光线与单面材质的物体交于物体背面而被吸收
                break;
            }
            else if (its_pre.HasEmission())
            { //按 BSDF 采样来自面光源的直接光照
                if (its.HarshLobe())
                    L += global_attenuation * its_pre.radiance() * local_attenuation;
                else
                {
                    Float pdf_direct = PdfEmitterDirect(its_pre, b_rec.wi),
                          weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
                    L += global_attenuation * weight_bsdf * its_pre.radiance() * local_attenuation;
                }
                break;
            }
            else
            { //按 BSDF 采样间接光照更新衰减系数
                global_attenuation *= local_attenuation;
            }

            if (depth > rr_depth_)
            { //处理俄罗斯轮盘赌算法
                global_attenuation /= pdf_rr_;
            }

            its = its_pre, wo = b_rec.wi, depth++;
        }
        return L;
    }
};

NAMESPACE_END(raytracer)