#pragma once

#include "../material.h"

NAMESPACE_BEGIN(simple_renderer)

class Conductor : public Material
{
public:
    /**
     * \brief 光滑的导体材质
     * \param id 材质id
     * \param mirror 是否是镜面（全反射）
     * \param eta 材质折射率的实部
     * \param k 材质折射率的虚部（消光系数）
     * \param ext_ior 外折射率
     * \param specular_reflectance 调节镜面反射分量。注意，对于物理真实感绘制，默认为 1，表示为空指针。
     */
    Conductor(const std::string &id,
              bool mirror,
              const Spectrum &eta,
              const Spectrum &k,
              Float ext_ior,
              std::unique_ptr<Texture> specular_reflectance = nullptr)
        : Material(id, MaterialType::kConductor),
          mirror_(mirror),
          eta_(eta / ext_ior),
          k_(k / ext_ior),
          specular_reflectance_(std::move(specular_reflectance))
    {
        if (mirror)
        {
            eta_ = Spectrum(0);
            k_ = Spectrum(1) / ext_ior;
        }
    }

    ///\brief 根据光线出射方向和表面法线方向，抽样光线入射方向
    BsdfSampling Sample(const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside, bool get_weight) const override
    {
        BsdfSampling bs;

        bs.wi = -Reflect(-wo, normal);
        bs.pdf = 1;

        if (get_weight)
        {
            bs.weight = mirror_ ? Spectrum(1) : FresnelConductor(bs.wi, normal, eta_, k_);
            if (specular_reflectance_)
            {
                if (texcoord != nullptr)
                    bs.weight *= specular_reflectance_->GetPixel(*texcoord);
                else
                    bs.weight *= specular_reflectance_->GetPixel(Vector2(0));
            }
        }

        return bs;
    }

    ///\brief 根据光线入射方向、出射方向和法线方向，计算 BSDF 权重
    Spectrum Eval(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        if (!SameDirection(wo, Reflect(wi, normal)))
            return Spectrum(0);

        auto albedo = mirror_ ? Spectrum(1) : FresnelConductor(wi, normal, eta_, k_);

        if (specular_reflectance_)
        {
            if (texcoord != nullptr)
                albedo *= specular_reflectance_->GetPixel(*texcoord);
            else
                albedo *= specular_reflectance_->GetPixel(Vector2(0));
        }

        return albedo;
    }

    ///\brief 根据光线入射方向和法线方向，计算光线从给定出射方向射出的概率
    Float Pdf(const Vector3 &wi, const Vector3 &wo, const Vector3 &normal, const Vector2 *texcoord, bool inside) const override
    {
        return SameDirection(wo, Reflect(wi, normal)) ? 1 : 0;
    }

    ///\brief 是否映射纹理
    bool TextureMapping() const override { return specular_reflectance_ && !specular_reflectance_->Constant(); }

private:
    bool mirror_;                                   //是否是镜面
    Spectrum eta_;                                  //材质相对折射率的实部
    Spectrum k_;                                    //材质相对折射率的虚部（消光系数）
    std::unique_ptr<Texture> specular_reflectance_; //调节镜面反射分量。对于物理真实感绘制，默认为 1，表示为空指针。
};

NAMESPACE_END(simple_renderer)