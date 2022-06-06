#pragma once

#include "accelerator.h"
#include "emitter.h"

enum IntegratorType
{
    kNoneIntegrator,
    kPath,
};

struct IntegratorInfo
{
    IntegratorType type;
    int max_depth;
    int rr_depth;

    IntegratorInfo()
        : type(kPath), max_depth(-1), rr_depth(5) {}

    IntegratorInfo(int max_depth, int rr_depth)
        : type(kPath), max_depth(max_depth), rr_depth(rr_depth) {}
};

//全局光照模型类
class Integrator
{
public:
    __device__ Integrator()
        : max_depth_(kUintMax), rr_depth_(5), pdf_rr_(0.9), emitter_num_(0), emitter_idx_list_(nullptr),
          scenebvh_(nullptr), shapebvh_list_(nullptr), env_map_(nullptr)
    {
    }

    __device__ void InitIntegrator(const IntegratorInfo &info, SceneBvh *scenebvh, ShapeBvh *shapebvh_list,
                                   uint *emitter_idx_list, uint emitter_num, EnvMap *env_map)
    {
        max_depth_ = info.max_depth;
        rr_depth_ = info.rr_depth;
        scenebvh_ = scenebvh;
        shapebvh_list_ = shapebvh_list;
        emitter_idx_list_ = emitter_idx_list;
        emitter_num_ = emitter_num;
        pdf_rr_ = 0.95;
        env_map_ = env_map;
    }

    __device__ vec3 Shade(const vec3 &eye_pos, const vec3 &look_dir, curandState *local_rand_state) const;

private:
    __device__ vec3 EmitterDirectArea(const Intersection &its, const vec3 &wo, curandState *local_rand_state) const;

    __device__ Float PdfEmitterDirect(const Intersection &its_pre, const vec3 &wi) const;

    int rr_depth_;            //最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
    Float pdf_rr_;            //递归地追踪光线俄罗斯轮盘赌的概率
    uint max_depth_;          //递归地追踪光线的最大深度
    uint emitter_num_;        //发光物体数量
    uint *emitter_idx_list_;  //发光物体索引
    SceneBvh *scenebvh_;      //场景层次包围盒
    ShapeBvh *shapebvh_list_; //物体层次包围盒
    EnvMap *env_map_;         //环境光映射
};

/**
 * @brief 着色
 *
 * @param eye_pos 观察点的坐标
 * @param look_dir 观察方向
 * @param local_rand_state 随机数生成器
 * @return 观察点来源于给定观察方向的辐射亮度
 */
__device__ vec3 Integrator::Shade(const vec3 &eye_pos, const vec3 &look_dir, curandState *local_rand_state) const
{
    auto L = vec3(0),          //着色结果
        attenuation = vec3(1); //光能因被物体吸收而衰减的系数
    uint depth = 0;            //光线溯源深度
    vec3 wo = -look_dir;
    auto its = Intersection(eye_pos);
    while (depth < max_depth_ && (depth <= rr_depth_ || curand_uniform(local_rand_state) < pdf_rr_))
    {
        bool harsh_lobe = its.HashLobe();
        //按发光物体表面积采样来自面光源的直接光照
        if (!harsh_lobe)
            L += attenuation * EmitterDirectArea(its, wo, local_rand_state);

        SamplingRecord b_rec = its.Sample(wo, RandomVec3(local_rand_state));
        if (!b_rec.valid)
            break;
        else
        {
            attenuation *= b_rec.attenuation / b_rec.pdf;
        }

        its = Intersection();
        bool hit_surface = scenebvh_->Intersect(Ray(b_rec.pos, -b_rec.wi), RandomVec2(local_rand_state), its);

        if (!hit_surface)
        {
            //按 BSDF 采样来自环境的直接光照
            if (env_map_ != nullptr)
                L += attenuation * env_map_->radiance(b_rec.wi) * b_rec.attenuation / b_rec.attenuation;
            break;
        }
        else
        { //光线来自景物表面
            if (its.absorb())
            { //光线与单面材质的物体交于物体背面而被吸收
                break;
            }
            else if (its.HasEmission())
            { //按 BSDF 采样来自面光源的直接光照
                if (harsh_lobe)
                    L += attenuation * its.radiance();
                else
                {
                    Float pdf_direct = PdfEmitterDirect(its, b_rec.wi),
                          weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
                    L += attenuation * weight_bsdf * its.radiance();
                }
                break;
            }
            else
            { //光线来自非发光物体的表面
                if (depth >= rr_depth_)
                { //处理俄罗斯轮盘赌算法
                    attenuation /= pdf_rr_;
                }
                wo = b_rec.wi,
                depth++;
            }
        }
    }
    return L;
}

/**
 * @brief 计算光线从某个发光物体向给定方向射出的概率（相对于光线出射后与物体表面交点处的立体角）
 *
 * @param its_pre 光线从发光物体射出的起点
 * @param wi 光线出射方向
 * @return 光线从某个发光物体向给定方向射出的概率（相对于光线出射后与物体表面交点处的立体角）
 */
__device__ Float Integrator::PdfEmitterDirect(const Intersection &its_pre, const vec3 &wi) const
{
    Float cos_theta_prime = myvec::dot(wi, its_pre.normal());
    if (cos_theta_prime < 0)
        return 0;
    Float pdf_area = its_pre.pdf_area() / emitter_num_,
          distance_sqr_ = its_pre.distance() * its_pre.distance();
    return pdf_area * distance_sqr_ / cos_theta_prime;
}

///\brief 按面积直接采样发光物体上一点，累计多重重要性采样下直接来自光源的辐射亮度
__device__ vec3 Integrator::EmitterDirectArea(const Intersection &its, const vec3 &wo, curandState *local_rand_state) const
{
    if (emitter_num_ == 0)
        return vec3(0);

    auto index = static_cast<uint>(curand_uniform(local_rand_state) * emitter_num_);
    if (index == emitter_num_)
        index--;
    index = emitter_idx_list_[index];
    auto its_pre = Intersection();
    shapebvh_list_[index].SampleP(its_pre, RandomVec3(local_rand_state));

    Float pdf_area = its_pre.pdf_area() / emitter_num_;

    vec3 d_vec = its.pos() - its_pre.pos();
    auto ray = Ray(its_pre.pos(), d_vec);
    auto its_test = Intersection();
    scenebvh_->Intersect(ray, RandomVec2(local_rand_state), its_test);
    if (its_test.distance() + kEpsilonDistance < d_vec.length())
        return vec3(0);

    vec3 wi = myvec::normalize(its.pos() - its_pre.pos());
    Float cos_theta_prime = myvec::dot(wi, its_pre.normal());
    if (cos_theta_prime < 0)
        return vec3(0);

    if (Perpendicular(-wi, its.normal()))
        return vec3(0);

    SamplingRecord b_rec = its.Eval(wi, wo);
    if (b_rec.pdf < kEpsilonPdf)
        return vec3(0);

    Float pdf_direct = pdf_area * d_vec.squared_length() / cos_theta_prime,
          weight_direct = MisWeight(pdf_direct, b_rec.pdf);
    return weight_direct * its_pre.radiance() * b_rec.attenuation / pdf_direct;
}

__global__ void InitIntegrator(IntegratorInfo info, SceneBvh *scenebvh, ShapeBvh *shapebvh_list, uint *emitter_idx_list,
                               uint emitter_num, EnvMap *env_map, Integrator *integrator)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        integrator->InitIntegrator(info, scenebvh, shapebvh_list, emitter_idx_list, emitter_num, env_map);
    }
}