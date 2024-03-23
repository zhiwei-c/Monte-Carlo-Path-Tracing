#pragma once

#include "../rtcore/scene.cuh"
#include "../tensor.cuh"
#include "../utils.cuh"
#include "bsdfs/bsdf.cuh"
#include "emitters/emitter.cuh"

namespace csrt
{
class Integrator
{
public:
    struct Data
    {
        // 根据俄罗斯轮盘赌算法的概率
        float pdf_rr = 0.95f;
        // 光线追踪开始使用俄罗斯轮盘赌算法判断是否终止的深度
        uint32_t depth_rr = 0;
        // 光线追踪的最大深度
        uint32_t depth_max = kMaxUint;

        // 面光源数量
        uint32_t num_area_light = 0;
        // 除面光源之外的其它光源数量
        uint32_t num_emitter = 0;
        // 太阳光源 ID
        uint32_t id_sun = kInvalidId;
        // 环境光照 ID
        uint32_t id_envmap = kInvalidId;

        // 场景中的所有BSDF
        Bsdf *bsdfs = nullptr;

        // 场景中的所有实例
        Instance *instances = nullptr;
        // 场景中所有实例按面积均匀抽样时的概率（面积的倒数）
        float *list_pdf_area_instance = nullptr;

        // 场景中的所有光源
        Emitter *emitters = nullptr;
        // 从面光源 ID 到相应实例 ID 的映射
        uint32_t *map_id_area_light_instance = nullptr;
        // 从实例 ID 到相应面光源 ID 的映射
        uint32_t *map_id_instance_area_light = nullptr;
        // 面光源抽样权重的累积分布函数
        float *cdf_area_light = nullptr;

        // 顶层加速结构
        TLAS *tlas = nullptr;
        // 从实例ID到相应BSDF ID的映射
        uint32_t *map_instance_bsdf = nullptr;
    };

    struct Info
    {
        // 根据俄罗斯轮盘赌算法的概率
        float pdf_rr = 0.95f;
        // 光线追踪开始使用俄罗斯轮盘赌算法判断是否终止的深度
        uint32_t depth_rr = 0;
        // 光线追踪的最大深度
        uint32_t depth_max = kMaxUint;
    };

    QUALIFIER_D_H Integrator();
    QUALIFIER_D_H Integrator(const Integrator::Data &data);

    QUALIFIER_D_H Vec3 Shade(const Vec3 &eye, const Vec3 &look_dir,
                             uint32_t *seed) const;

private:
    QUALIFIER_D_H Vec3 EvaluateDirectLight(const Hit &hit, const Vec3 &wo,
                                           uint32_t *seed) const;
    QUALIFIER_D_H BsdfSampleRec EvaluateRay(const Vec3 &wi, const Vec3 &wo,
                                              const Hit &hit, Bsdf *bsdf) const;
    QUALIFIER_D_H BsdfSampleRec SampleRay(const Vec3 &wo, const Hit &hit,
                                            Bsdf *bsdf, uint32_t *seed) const;

    uint32_t size_cdf_area_light_;
    float pdf_rr_rcp_;
    Data data_;
};

} // namespace csrt
