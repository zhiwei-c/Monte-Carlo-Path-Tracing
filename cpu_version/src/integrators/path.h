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
    PathIntegrator(int max_depth, int rr_depth) : Integrator(IntegratorType::kPath, max_depth, rr_depth) {}

    ///\brief 着色
    ///\param eye_pos 观察点的位置
    ///\param look_dir 观察方向
    ///\return 观察点来源于给定观察方向的辐射亮度
    Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const override
    {
        auto L = Spectrum(0),                      //着色结果
            global_attenuation = Spectrum(1);      //光能因被物体吸收而衰减的系数
        size_t depth = 1;                          //光线溯源深度
        Vector3 wo = -look_dir;                    //当前出射光线方向
        auto its = Intersection(eye_pos, nullptr); //当前散射点
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
                global_attenuation *= b_rec.attenuation / b_rec.pdf;

            its = Intersection();
            bool hit_surface = this->bvh_ && this->bvh_->Intersect(Ray(b_rec.pos, -b_rec.wi), its), //光线是否可能源于景物表面
                inside = false;                                                                     //当前散射点是否在某个物体的内部
            Medium *medium_now = global_medium_;                                                    //当前散射点是否在某个物体内部
            auto max_distance = std::numeric_limits<Float>::infinity();                             //当前散射点距离可能的景物表面光线源头的距离
            if (hit_surface)
            {
                max_distance = its.distance();
                if (depth != 1)
                    inside = its.Inner(b_rec.wi);
                if (inside)
                    medium_now = its.medium();
            }

            if (medium_now != nullptr)
            { //当前散射点在参与介质之中
                Float actual_distance = 0, pdf_scatter = 0;
                auto medium_attenuation = Spectrum(0);
                bool scattered = medium_now->SampleDistance(max_distance, actual_distance, pdf_scatter, medium_attenuation);
                global_attenuation *= medium_attenuation / pdf_scatter;
                if (scattered)
                { //光线在传播时发生了散射，实际上来源于更近的地方
                    its = Intersection(b_rec.pos - actual_distance * b_rec.wi, medium_now);
                    if (!inside)
                    { //只对外部光线计数深度，确保物体内部的光线散射出去了
                        if (depth > rr_depth_)
                        { //处理俄罗斯轮盘赌算法
                            global_attenuation /= pdf_rr_;
                        }
                        depth++;
                    }
                    wo = b_rec.wi;
                    continue;
                }
            }

            if (!hit_surface)
            { //没有散射，光线来自环境
                if (envmap_ != nullptr)
                    L += envmap_->radiance(-b_rec.wi) * global_attenuation;
                break;
            }
            else
            { //没有散射，光线来自景物表面
                if (its.absorb())
                { //没有散射，光线与单面材质的物体交于物体背面而被吸收
                    break;
                }
                else if (its.HasEmission())
                { //没有散射，按 BSDF 采样来自面光源的直接光照
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
                else
                { //没有散射，光线来自非发光物体的表面
                    if (depth > rr_depth_)
                    { //处理俄罗斯轮盘赌算法
                        global_attenuation /= pdf_rr_;
                    }
                    depth++;
                    wo = b_rec.wi;
                }
            }
        }
        return L;
    }
};
NAMESPACE_END(raytracer)