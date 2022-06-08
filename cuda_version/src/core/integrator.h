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
                                   uint *emitter_idx_list, uint emitter_num, EnvMap *env_map);

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

__global__ inline void InitIntegrator(IntegratorInfo info, SceneBvh *scenebvh, ShapeBvh *shapebvh_list, uint *emitter_idx_list,
                                      uint emitter_num, EnvMap *env_map, Integrator *integrator)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        integrator->InitIntegrator(info, scenebvh, shapebvh_list, emitter_idx_list, emitter_num, env_map);
    }
}