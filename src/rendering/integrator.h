#pragma once

#include <memory>

#include "../modeling/envmap.h"
#include "../utils/accelerator/bvh_accel.h"
#include "../modeling/scene.h"

NAMESPACE_BEGIN(simple_renderer)

//着色算法积分器基类
class Integrator
{
public:
    ///\brief 着色算法积分器类
    Integrator(Scene *scene)
        : envmap_(scene->envmap()), bvh_(nullptr)
    {
        emitters_.clear();
        emit_area_ = 0;
        for (auto &shape : scene->shapes())
        {
            if (shape->HasEmission())
            {
                emitters_.push_back(shape);
                emit_area_ += shape->area();
            }
        }
        if (!scene->shapes().empty())
            bvh_ = std::make_unique<BvhAccel>(scene->shapes());
    }

    /**
     * \brief 着色
     * \param eye_pos 观察点的坐标
     * \param look_dir 观察方向
     * \return 观察点来源于给定观察方向的辐射亮度
     */
    virtual Spectrum Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const = 0;

protected:
    std::unique_ptr<BvhAccel> bvh_; //层次包围盒
    std::vector<Shape *> emitters_; //包含的发光物体
    Float emit_area_;               //发光物体的总表面积
    Envmap *envmap_;                //用于绘制的天空盒
};

NAMESPACE_END(simple_renderer)