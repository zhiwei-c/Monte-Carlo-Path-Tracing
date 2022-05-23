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
    __device__ Integrator() : max_depth_(kUintMax),
                              rr_depth_(5),
                              pdf_rr_(0.9),
                              emitter_num_(0),
                              emitter_idx_list_(nullptr),
                              scenebvh_(nullptr),
                              shapebvh_list_(nullptr),
                              env_map_(nullptr) {}

    __device__ void InitIntegrator(const IntegratorInfo &info,
                                   SceneBvh *scenebvh,
                                   ShapeBvh *shapebvh_list,
                                   uint *emitter_idx_list,
                                   uint emitter_num,
                                   EnvMap *env_map)
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
    int rr_depth_;            //最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
    Float pdf_rr_;            //递归地追踪光线俄罗斯轮盘赌的概率
    uint max_depth_;          //递归地追踪光线的最大深度
    uint emitter_num_;        //发光物体数量
    uint *emitter_idx_list_;  //发光物体索引
    SceneBvh *scenebvh_;      //场景层次包围盒
    ShapeBvh *shapebvh_list_; //物体层次包围盒
    EnvMap *env_map_;         //环境光映射

    __device__ bool EmitterDirectArea(const Intersection &its, const vec3 &wo, const vec3 &history_attenuation, curandState *local_rand_state, vec3 &L) const;

    __device__ Float PdfEmitterDirect(const Intersection &its_pre, const vec3 &wi) const;
};

/**
 * @brief 着色
 *
 * @param eye_pos 观察点的坐标
 * @param look_dir 观察方向
 * @param local_rand_state 随机数生成器
 * @return 观察点来源于给定观察方向的辐射亮度
 */
__device__ vec3 Integrator::Shade(const vec3 &eye_pos,
                                  const vec3 &look_dir,
                                  curandState *local_rand_state) const
{
    auto its = Intersection();

    //原初光线源于环境
    if (!scenebvh_->Intersect(Ray(eye_pos, look_dir), RandomVec2(local_rand_state), its))
        return env_map_ ? env_map_->radiance(-look_dir) : vec3(0);

    ///单面材质物体的背面，只吸收而不反射或折射光线
    if (its.absorb())
        return vec3(0);

    //原初光线源于发光物体
    if (its.HasEmission())
        return its.radiance();

    vec3 wo = -look_dir;
    auto its_pre = Intersection();
    uint depth = 1;            //光线溯源深度
    auto L = vec3(0),          //着色结果
        attenuation = vec3(1); //光能因被物体吸收而衰减的系数
    //迭代地溯源光线
    while (depth < max_depth_ && (depth <= rr_depth_ || curand_uniform(local_rand_state) < pdf_rr_))
    {
        //按发光物体表面积采样来自面光源的直接光照
        if (!its.HashLobe())
            EmitterDirectArea(its, wo, attenuation, local_rand_state, L);

        BsdfSampling b_rec = BsdfSampling();
        b_rec.wo = wo;
        its.Sample(b_rec, RandomVec3(local_rand_state));
        if (!b_rec.valid)
            break;
        Float cos_theta = abs(myvec::dot(b_rec.wi, its.normal()));
        its_pre = Intersection();

        //按 BSDF 采样来自环境的直接光照
        if (!scenebvh_->Intersect(Ray(its.pos(), -b_rec.wi), RandomVec2(local_rand_state), its_pre))
        {
            if (env_map_ != nullptr)
                L += attenuation * env_map_->radiance(b_rec.wi) * b_rec.attenuation * cos_theta / b_rec.attenuation;
            break;
        }

        //光线与单面材质的物体交于物体背面而被吸收
        if (its_pre.absorb())
            break;

        //按 BSDF 采样来自面光源的直接光照
        if (its_pre.HasEmission())
        {
            if (its.HashLobe())
                L += attenuation * its_pre.radiance() * b_rec.attenuation * cos_theta / b_rec.pdf;
            else
            {
                Float pdf_direct = PdfEmitterDirect(its_pre, b_rec.wi),
                      weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
                L += attenuation * weight_bsdf * its_pre.radiance() * b_rec.attenuation * cos_theta / b_rec.pdf;
            }
            break;
        }

        //按 BSDF 采样间接光照更新衰减系数
        attenuation *= b_rec.attenuation * cos_theta / b_rec.pdf;
        if (depth >= rr_depth_)
            attenuation /= pdf_rr_;

        its = its_pre,
        wo = b_rec.wi,
        depth++;
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
__device__ Float Integrator::PdfEmitterDirect(const Intersection &its_pre,
                                              const vec3 &wi) const
{
    Float cos_theta_prime = myvec::dot(wi, its_pre.normal());
    if (cos_theta_prime < 0)
        return 0;
    Float pdf_area = its_pre.pdf_area() / emitter_num_,
          distance_sqr_ = its_pre.distance() * its_pre.distance();
    return pdf_area * distance_sqr_ / cos_theta_prime;
}

///\brief 按面积直接采样发光物体上一点，累计多重重要性采样下直接来自光源的辐射亮度
__device__ bool Integrator::EmitterDirectArea(const Intersection &its,
                                              const vec3 &wo,
                                              const vec3 &history_attenuation,
                                              curandState *local_rand_state,
                                              vec3 &L) const
{
    if (emitter_num_ == 0)
        return false;

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
        return false;

    vec3 wi = myvec::normalize(its.pos() - its_pre.pos());
    Float cos_theta_prime = myvec::dot(wi, its_pre.normal());
    if (cos_theta_prime < 0)
        return false;

    if (Perpendicular(-wi, its.normal()))
        return false;

    Float local_pdf = its.Pdf(wi, wo);
    if (local_pdf < kEpsilonPdf)
        return false;

    Float pdf_direct = pdf_area * d_vec.squared_length() / cos_theta_prime,
          weight_direct = MisWeight(pdf_direct, local_pdf),
          cos_theta = abs(myvec::dot(wi, its.normal()));
    L += history_attenuation * weight_direct * its_pre.radiance() * its.Eval(wi, wo) * cos_theta / pdf_direct;
    return true;
}

__global__ void InitIntegrator(IntegratorInfo info,
                               SceneBvh *scenebvh,
                               ShapeBvh *shapebvh_list,
                               uint *emitter_idx_list,
                               uint emitter_num,
                               EnvMap *env_map,
                               Integrator *integrator)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        integrator->InitIntegrator(info, scenebvh, shapebvh_list, emitter_idx_list, emitter_num, env_map);
    }
}