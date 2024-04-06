#pragma once

#include "../global.hpp"
#include "../core/camera.hpp"
#include "../core/intersection.hpp"

NAMESPACE_BEGIN(raytracer)

//全局光照模型的类型
enum class IntegratorType
{
    kPath,    //基本的路径跟踪算法
    kVolPath, //支持体绘制的路径跟踪算法
    kBdpt,    //双向路径跟踪算法
};

//全局光照模型，计算绘制方程定积分的方法
class Integrator
{
public:
    Integrator(IntegratorType type, int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
               const std::vector<Emitter *> &emitters, size_t shape_num);
    virtual ~Integrator() {}

    virtual std::vector<float> Shade(const Camera &camera) const;
    virtual dvec3 Shade(const dvec3 &eye, const dvec3 &look_dir, Sampler *sampler) const = 0;

protected:
    virtual dvec3 SampleAreaLightsDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const;
    virtual dvec3 SampleOtherEmittersDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const;
    double PdfAreaLight(const Intersection &its_light, const dvec3 &wi) const;
    double PdfEnvmap(const dvec3 &wi) const;
    bool Visible(const Intersection &a, const Intersection &b, Sampler *sampler) const;

    IntegratorType type_;                       //全局光照模型的类型
    bool hide_emitters_;                        //隐藏直接可见的光源
    size_t max_depth_;                          //递归地追踪光线的最大深度
    size_t rr_depth_;                           //最小的光线追踪深度，超过该深度后进行俄罗斯轮盘赌抽样控制光线追踪深度
    Accelerator *accelerator_;                  //场景几何加速结构，用于求取光线和景物表面之间的交点
    double area_light_num_rcp_;                 //场景中面光源数量的倒数
    double no_emitter_num_rcp_;                 //场景中非发光物体数量的倒数
    double pdf_rr_ = 0.8;                       //俄罗斯轮盘赌算法概率
    Emitter *envmap_;                           //环境映射、天空
    std::vector<Emitter *> area_lights_;        //面光源
    std::vector<Emitter *> point_lights_;       //点光源、聚光灯
    std::vector<Emitter *> directional_lights_; //方向光、太阳

private:
    dvec3 SampleEnvmapDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const;
};

NAMESPACE_END(raytracer)