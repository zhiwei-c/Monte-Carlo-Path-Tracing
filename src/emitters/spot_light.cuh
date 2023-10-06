#pragma once

#include "emitter.cuh"

class SpotLight : public Emitter
{
public:
    QUALIFIER_DEVICE SpotLight(const uint64_t id, const Emitter::Info::Data::Spot &data);

    QUALIFIER_DEVICE bool GetRadiance(const Vec3 &origin, const Accel *accel, Bsdf **bsdf_buffer,
                                      Texture **texture_buffer, const float *pixel_buffer,
                                      uint64_t *seed, Vec3 *radiance, Vec3 *wi) const override;

private:
    uint64_t id_texture_;        // 纹理
    float cutoff_angle_;         // 截光角（弧度制）
    float cos_cutoff_angle_;     // 截光角的余弦
    float uv_factor_;            // 用于计算纹理坐标的系数
    float cos_beam_width_;       // 截光角中光线不衰减部分的余弦
    float transition_width_rcp_; // 截光角中光线衰减部分角度的倒数（弧度制）
    Vec3 intensity_;             // 辐射强度
    Vec3 position_world_;        // 世界空间下的位置
    Mat4 to_local_;              // 从世界坐标系转换到局部坐标系的变换矩阵
};