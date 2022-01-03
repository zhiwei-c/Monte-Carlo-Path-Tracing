#pragma once

#include "../utils/accelerator/aabb.h"
#include "../material/material.h"

NAMESPACE_BEGIN(simple_renderer)

//光线与面片交点类
class Intersection
{
public:
    ///\brief 光线与物体模型面片交点，光线与物体没有相交
    Intersection()
        : valid_(false),
          pos_(Vector3(0)),
          normal_(Vector3(0)),
          inside_(false),
          distance_(INFINITY),
          texcoord_(Vector2(0)),
          material_(nullptr) {}

    /**
     * \brief 光线与物体模型面片交点，光线与物体相交
     * \param pos 空间坐标
     * \param normal 法线
     * \param texcoord 纹理坐标
     * \param inside 法线是否朝内
     * \param distance 光线起点与交点间的距离
     * \param material 交点处物体表面的材质
     */
    Intersection(const Vector3 &pos,
                 const Vector3 &normal,
                 const Vector2 &texcoord,
                 bool inside,
                 Float distance,
                 Material *material)
        : valid_(true),
          pos_(pos),
          normal_(normal),
          inside_(inside),
          distance_(distance),
          texcoord_(texcoord),
          material_(material) {}

    /**
     * \brief 交点处给定光线出射方向，采样入射方向
     * \param wo 给定的光线出射方向
     * \return 采样得到的光线入射方向
     */
    std::pair<Vector3, BsdfSamplingType> SampleWi(const Vector3 &wo) const
    {
        auto one_side = glm::dot(wo, normal_) > 0; //光线与交点法线是否同侧
        auto normal = one_side ? normal_ : -normal_;
        auto inside = one_side ? inside_ : !inside_;

        return material_->TextureMapping()
                   ? material_->Sample(wo, normal, &texcoord_, inside)
                   : material_->Sample(wo, normal, nullptr, inside);
    }

    /**
     * \brief 交点处光线从给定方向射入后，从给定方向射出，
     *      计算BSDF（bidirectional scattering distribution function，双向散射分别函数）系数
     * \param wi 给定的光线射入方向
     * \param wo 给定的光线射出方向
     * \return 算得的BSDF系数
     */
    Vector3 Eval(const Vector3 &wi, const Vector3 &wo, const BsdfSamplingType &bsdf_sampling_type) const
    {
        auto one_side = glm::dot(wi, normal_) < 0; //入射光线与法线是否同侧
        auto normal = one_side ? normal_ : -normal_;
        auto inside = one_side ? inside_ : !inside_;

        if (material_->TextureMapping())
            return material_->Eval(wi, wo, normal, &texcoord_, inside, bsdf_sampling_type);
        else
            return material_->Eval(wi, wo, normal, nullptr, inside, bsdf_sampling_type);
    }

    /**
     * \brief 交点处光线从给定方向射入后，从给定方向射出的概率
     * \param wi 给定的光线射入方向
     * \param wo 给定的光线射出方向
     * \return 算得的概率
     */
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const BsdfSamplingType &bsdf_sampling_type) const
    {

        auto one_side = glm::dot(wi, normal_) < 0; //入射光线与法线是否同侧
        auto normal = one_side ? normal_ : -normal_;
        auto inside = one_side ? inside_ : !inside_;

        return (material_->TextureMapping())
                   ? material_->Pdf(wi, wo, normal, &texcoord_, inside, bsdf_sampling_type)
                   : material_->Pdf(wi, wo, normal, nullptr, inside, bsdf_sampling_type);
    }

    ///\return 交点的位置
    Vector3 pos() const { return pos_; }

    ///\return 交点处的物体表面的法线
    Vector3 normal() const { return normal_; }

    ///\return 光线与物体的相交是否发生
    bool valid() const { return valid_; }

    ///\return 从光线起点到该交点的距离
    Float distance() const { return distance_; }

    ///\return 交点处的物体表面是否发光
    bool HasEmission() const { return material_->HasEmission(); }

    ///\return 交点处的物体表面的辐射亮度
    Vector3 radiance() const { return material_->radiance(); }

private:
    bool valid_;         //光线与物体的相交是否发生
    bool inside_;        //交点处法线是否朝内
    Vector3 pos_;        //交点空间坐标
    Vector3 normal_;     //交点法线
    Vector2 texcoord_;   //交点纹理坐标
    Float distance_;     //从光线起点到该交点的距离
    Material *material_; //交点面片对应的材质
};

NAMESPACE_END(simple_renderer)