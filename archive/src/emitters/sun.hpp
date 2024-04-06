#pragma once

#include "emitter.hpp"
#include "../core/intersection.hpp"

NAMESPACE_BEGIN(raytracer)

//太阳
class Sun : public Emitter
{
public:
    Sun(const dvec3 &sun_direction, double turbidity, int resolution, double sun_scale, double sun_radius_scale);
    ~Sun();

    SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                          Accelerator *accelerator) const override;

    dvec3 radiance(const dvec3 &position, const dvec3 &wi) const override;

private:
    int resolution_;                              //环境映射纹理的水平像素数
    double normalization_;                        //计算方向光概率的归一化系数
    std::vector<double> cdf_;                     //方向光的累积分布函数
    std::vector<Emitter *> directionals_;         //代表太阳光的遥远的方向光
    std::vector<std::pair<int, int>> sun_pixels_; //太阳在环境映射纹理中占有的像素
    std::vector<dvec3> pixel_colors_;             //太阳在环境映射纹理中占有像素的颜色
};

NAMESPACE_END(raytracer)