#pragma once

#include <utility>
#include <vector>

#include "emitter.hpp"
#include "../core/intersection.hpp"

NAMESPACE_BEGIN(raytracer)

//环境光
class Envmap : public Emitter
{
public:
    Envmap(Texture *background, double scale, dmat4 to_world);

    SamplingRecord Sample(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler,
                          Accelerator *accelerator) const override;
    double Pdf(const dvec3 &look_dir) const override;
    dvec3 radiance(const dvec3 &position, const dvec3 &wi) const override;

private:
    bool use_importance_sampling_;              //是否根据像素亮度进行重要抽样
    Texture *background_;                       //环境纹理
    int height_;                                //纹理的垂直像素数
    int width_;                                 //纹理的水平像素数
    double height_rcp_;                         //纹理垂直像素数的倒数
    double width_rcp_;                          //纹理水平像素数的倒数
    double scale_;                              //颜色的放缩系数
    double normalization_;                      //计算像素概率的归一化系数
    dmat4 to_local_;                            //从世界空间到局部空间的变换矩阵
    dmat4 to_world_;                            //从局部空间到世界空间的变换矩阵
    std::vector<double> cdf_rows_;              //正弦加权的、像素行的累积分布函数
    std::vector<double> weight_rows_;           //各个像素行的权重
    std::vector<std::vector<double>> cdf_cols_; //像素列的累积分布函数
};

NAMESPACE_END(raytracer)