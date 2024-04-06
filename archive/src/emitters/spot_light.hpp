#pragma once

#include "emitter.hpp"
#include "../core/intersection.hpp"

NAMESPACE_BEGIN(raytracer)

//聚光灯
class SpotLight : public Emitter
{
public:
    SpotLight(const dvec3 &intensity, double cutoff_angle, double beam_width, Texture *texture, const dmat4 &to_world);

    SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler* sampler,
                          Accelerator *accelerator) const override;

private:
    bool FallOffCurve(const dvec3 &local_dir, dvec3 *value) const;

    Texture *texture_;            //纹理
    double cutoff_angle_;         //截光角（弧度制）
    double cos_cutoff_angle_;     //截光角的余弦
    double uv_factor_;            //用于计算纹理坐标的系数
    double cos_beam_width_;       //截光角中光线不衰减部分的余弦
    double transition_width_rcp_; //截光角中光线衰减部分角度的倒数（弧度制）
    dvec3 intensity_;             //辐射强度
    dvec3 position_world_;        //世界空间下的位置
    dmat4 to_local_;              //从世界坐标系转换到局部坐标系的变换矩阵
};

NAMESPACE_END(raytracer)