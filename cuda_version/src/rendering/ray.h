#pragma once

#include <map>

#include "../utils/global.h"

class Ray
{
public:
    /**
     * \brief 光线
     * \param origin 起点
     * \param dir 方向
     */
    __device__ Ray(const vec3 &origin, const vec3 &dir) : origin_(origin), dir_(myvec::normalize(dir))
    {
        dir_inv_ = vec3(1 / dir_.x, 1 / dir_.y, 1 / dir_.z);
    }

    ///\return 光线方向
    __device__ vec3 dir() const { return dir_; }

    ///\return 光线方向的倒数
    __device__ vec3 dir_inv() const { return dir_inv_; }

    ///\return 光线起点
    __device__ vec3 origin() const { return origin_; }

private:
    vec3 origin_;  //光线起点
    vec3 dir_;     //光线方向
    vec3 dir_inv_; //光线方向的倒数
};

/**
 * \brief 根据光线入射方向与表面法线方向，计算光线完美镜面反射方向
 * \param wi 光线入射方向
 * \param normal 表面法线方向（注意：已处理为与光线入射方向夹角大于90度）
 * \return 完美镜面反射方向
 */
__device__ inline vec3 Reflect(const vec3 &wi, vec3 normal)
{
    return myvec::normalize(wi - static_cast<Float>(2 * myvec::dot(wi, normal)) * normal);
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算光线完美折射方向
 * \param wi 光线入射方向；
 * \param normal 表面法线方向；（注意：已处理为与光线入射方向夹角大于90度）
 * \param eta_inv 相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
 * \return 光线完美折射方向
 */
__device__ inline vec3 Refract(const vec3 &wi, const vec3 &normal, Float eta_inv)
{
    auto cos_theta_i = abs(myvec::dot(wi, normal));
    auto k = 1 - eta_inv * eta_inv * (1 - cos_theta_i * cos_theta_i);
    return (k < 0) ? vec3(0) : myvec::normalize((eta_inv * wi + (eta_inv * cos_theta_i - sqrt(k)) * normal));
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算菲涅尔系数；
 * \param wi 光线入射方向
 * \param normal 表面法线方向（注意：已处理为与光线入射方向夹角大于90度）
 * \param eta_inv 相对折射率的倒数，即光线入射侧介质折射率与透射侧介质折射率之比
 * \return 菲涅尔系数。
 */
__device__ inline Float Fresnel(const vec3 &wi, const vec3 &normal, Float eta_inv)
{
    auto cos_theta_i = abs(myvec::dot(wi, normal));
    auto cos_theta_t_2 = 1.0 - eta_inv * eta_inv * (1.0 - cos_theta_i * cos_theta_i);

    if (cos_theta_t_2 <= 0)
    {
        return 1;
    }
    else
    {
        auto cos_theta_t = sqrt(cos_theta_t_2);
        auto Rs_sqrt = (eta_inv * cos_theta_i - cos_theta_t) / (eta_inv * cos_theta_i + cos_theta_t),
             Rp_sqrt = (cos_theta_i - eta_inv * cos_theta_t) / (cos_theta_i + eta_inv * cos_theta_t);
        return (Rs_sqrt * Rs_sqrt + Rp_sqrt * Rp_sqrt) / 2;
    }
}

/**
 * \brief 根据光线入射方向、表面法线方向与相对折射率，计算导体的菲涅尔系数；
 * \param wi 光线入射方向
 * \param normal 表面法线方向
 * \param eta_r 相对折射率的实部
 * \param eta_i 相对折射率的虚部（消光系数）
 * \return 菲涅尔系数
 */
__device__ inline vec3 FresnelConductor(const vec3 &wi, const vec3 &normal, const vec3 &eta_r, const vec3 &eta_i)
{
    auto cos_theta_i = myvec::dot(-wi, normal);
    auto cos_theta_i_2 = cos_theta_i * cos_theta_i,
         sin_theta_i_2 = 1.0 - cos_theta_i_2,
         sin_theta_i_4 = sin_theta_i_2 * sin_theta_i_2;

    auto temp_1 = eta_r * eta_r - eta_i * eta_i - sin_theta_i_2;

    auto a_2_pb_2 = temp_1 * temp_1 + static_cast<Float>(4) * eta_i * eta_i * eta_r * eta_r;
    for (int i = 0; i < 3; i++)
    {
        a_2_pb_2[i] = sqrt(glm::max(0.0, a_2_pb_2[i]));
    }

    auto a = static_cast<Float>(.5) * (a_2_pb_2 + temp_1);
    for (int i = 0; i < 3; i++)
    {
        a[i] = sqrt(glm::max(0.0, a[i]));
    }

    auto term_1 = a_2_pb_2 + cos_theta_i_2,
         term_2 = 2.0 * cos_theta_i * a;

    auto r_s = (term_1 - term_2) / (term_1 + term_2);

    auto term_3 = a_2_pb_2 * cos_theta_i_2 + sin_theta_i_4,
         term_4 = term_2 * sin_theta_i_2;

    auto r_p = r_s * (term_3 - term_4) / (term_3 + term_4);

    return static_cast<Float>(0.5) * (r_s + r_p);
}

/**
 * \brief Computes the diffuse unpolarized Fresnel reflectance of a dielectric
 *		material (sometimes referred to as "Fdr").
 *		This value quantifies what fraction of diffuse incident illumination
 *		will, on average, be reflected at a dielectric material boundary
 * \param eta Relative refraction coefficient
 * \return F, the unpolarized Fresnel coefficient.
 */
__device__ inline Float FresnelDiffuseReflectance(Float eta)
{
    if (eta < 1)
    {
        /* Fit by Egan and Hilgeman (1973). Works reasonably well for
            "normal" IOR values (<2).
            Max rel. error in 1.0 - 1.5 : 0.1%
            Max rel. error in 1.5 - 2   : 0.6%
            Max rel. error in 2.0 - 5   : 9.5%
        */
        return -1.4399 * (eta * eta) + 0.7099 * eta + 0.6681 + 0.0636 / eta;
    }
    else
    {
        /* Fit by d'Eon and Irving (2011)

            Maintains a good accuracy even for unrealistic IOR values.

            Max rel. error in 1.0 - 2.0   : 0.1%
            Max rel. error in 2.0 - 10.0  : 0.2%
        */
        auto inv_eta = 1.0 / eta,
             inv_eta_2 = inv_eta * inv_eta,
             inv_eta_3 = inv_eta_2 * inv_eta,
             inv_eta_4 = inv_eta_3 * inv_eta,
             inv_eta_5 = inv_eta_4 * inv_eta;
        return 0.919317 - 3.4793 * inv_eta + 6.75335 * inv_eta_2 - 7.80989 * inv_eta_3 + 4.98554 * inv_eta_4 - 1.36881 * inv_eta_5;
    }
}

// 各种电介质材质（dielectric）的折射率（refractive index）
const std::map<std::string, Float> IOR{
    //真空
    {"vacuum", 1.0f},
    //溴
    {"bromine", 1.661f},
    //氦
    {"helium", 1.000036f},
    //水冰
    {"water ice", 1.31f},
    {"water-ice", 1.31f},
    //氢
    {"hydrogen", 1.000132f},
    //熔融石英
    {"fused quartz", 1.458f},
    {"fused-quartz", 1.458f},
    //空气
    {"air", 1.000277f},
    //派热克斯玻璃（一种耐热玻璃）
    {"pyrex", 1.470f},
    //二氧化碳
    {"carbon dioxide", 1.00045f},
    {"carbon-dioxide", 1.00045f},
    //丙烯酸玻璃
    {"acrylic glass", 1.49f},
    {"acrylic-glass", 1.49f},
    //水
    {"water", 1.3330f},
    //聚丙烯
    {"polypropylene", 1.49f},
    //丙酮
    {"acetone", 1.36f},
    // BK7玻璃
    {"bk7", 1.5046f},
    //乙醇
    {"ethanol", 1.361f},
    //氯化钠
    {"sodium chloride", 1.544f},
    {"sodium-chloride", 1.544f},
    //四氯化碳
    {"carbon tetrachloride", 1.461f},
    {"carbon-tetrachloride", 1.461f},
    //琥珀
    {"amber", 1.55f},
    //甘油
    {"glycerol", 1.4729f},
    //聚对苯二甲酸乙二醇酯
    {"pet", 1.5750f},
    //苯
    {"benzene", 1.501f},
    //金刚石
    {"diamond", 2.419f},
    //硅油
    {"silicone oil", 1.52045f},
    {"silicone-oil", 1.52045f},
};

//各种导体材质（conductor）的折射率（refractive index）
const std::map<std::string, vec3> IOR_eta{
    //无定形碳 Amorphous carbon
    {"a-C", vec3(2.93785f, 2.22242f, 1.96400f)},
    //银 Silver
    {"Ag", vec3(0.15494f, 0.11648f, 0.13809f)},
    //铝 Aluminium
    {"Al", vec3(1.65394f, 0.87850f, 0.52012f)},
    //氧化铝 Aluminium oxide
    {"Al2O3", vec3(2.12296f, 1.67556f, 1.61569f)},
    //立方砷化铝 Cubic aluminium arsenide
    {"AlAs", vec3(3.59737f, 3.22615f, 2.21235f)},
    {"AlAs_palik", vec3(3.59737f, 3.22615f, 2.21235f)},
    //立方锑化铝 Cubic aluminium antimonide
    {"AlSb", vec3(4.70830f, 4.05196f, 4.57419f)},
    {"AlSb_palik", vec3(4.70830f, 4.05196f, 4.57419f)},
    //金 Gold
    {"Au", vec3(0.14282f, 0.37414f, 1.43944f)},
    //多晶铍 Polycrystalline beryllium
    {"Be", vec3(4.17618f, 3.17830f, 2.77819f)},
    {"Be_palik", vec3(4.17618f, 3.17830f, 2.77819f)},
    //铬 Chromium
    {"Cr", vec3(4.36041f, 2.91052f, 1.65119f)},
    //立方碘化铯 Cubic caesium iodide
    {"CsI", vec3(2.14035f, 1.69870f, 1.65890f)},
    {"CsI_palik", vec3(2.14035f, 1.69870f, 1.65890f)},
    //铜 Copper
    {"Cu", vec3(0.19999f, 0.92209f, 1.09988f)},
    {"Cu_palik", vec3(0.21221f, 0.91805f, 1.10007f)},
    //氧化亚铜 Cuprous oxide
    {"Cu2O", vec3(3.54426f, 2.94365f, 2.74619f)},
    {"Cu2O_palik", vec3(3.54426f, 2.94365f, 2.74619f)},
    //氧化铜 Cupric oxide
    {"CuO", vec3(3.24062f, 2.44261f, 2.20556f)},
    {"CuO_palik", vec3(3.24062f, 2.44261f, 2.20556f)},
    //立方金刚石 Cubic diamond
    {"d-C", vec3(2.90008f, 2.29322f, 2.22275f)},
    {"d-C_palik", vec3(2.90008f, 2.29322f, 2.22275f)},
    //铁 Iron
    {"Fe", vec3(2.76404f, 1.95417f, 1.62766f)},
    //氮化镓 Gallium nitride
    {"GaN", vec3(2.86583f, 2.29184f, 2.29848f)},
    //锗 Germanium
    {"Ge", vec3(7.05832f, 4.78868f, 3.57814f)},
    //汞 Mercury
    {"Hg", vec3(2.39384f, 1.43697f, 0.90762f)},
    {"Hg_palik", vec3(2.39384f, 1.43697f, 0.90762f)},
    //碲化汞 Mercury telluride
    {"HgTe", vec3(4.76940f, 3.22413f, 2.65439f)},
    {"HgTe_palik", vec3(4.76940f, 3.22413f, 2.65439f)},
    //铱 Iridium
    {"Ir", vec3(3.07986f, 2.07777f, 1.61446f)},
    {"Ir_palik", vec3(3.07986f, 2.07777f, 1.61446f)},
    //多晶钾 Polycrystalline potassium
    {"K", vec3(0.06391f, 0.04631f, 0.03810f)},
    {"K_palik", vec3(0.06391f, 0.04631f, 0.03810f)},
    //锂 Lithium
    {"Li", vec3(0.26525f, 0.19519f, 0.22045f)},
    {"Li_palik", vec3(0.26525f, 0.19519f, 0.22045f)},
    //氧化镁 Magnesium oxide
    {"MgO", vec3(2.08521f, 1.64721f, 1.59150f)},
    {"MgO_palik", vec3(2.08521f, 1.64721f, 1.59150f)},
    //钼 Molybdenum
    {"Mo", vec3(4.47417f, 3.51799f, 2.77018f)},
    {"Mo_palik", vec3(4.47417f, 3.51799f, 2.77018f)},
    //钠 Sodium
    {"Na", vec3(0.06014f, 0.05602f, 0.06186f)},
    {"Na_palik", vec3(0.06014f, 0.05602f, 0.06186f)},
    //铌 Niobium
    {"Nb", vec3(3.41288f, 2.78427f, 2.39051f)},
    {"Nb_palik", vec3(3.41288f, 2.78427f, 2.39051f)},
    //镍 Nickel
    {"Ni", vec3(2.36225f, 1.65983f, 1.46395f)},
    {"Ni_palik", vec3(2.36225f, 1.65983f, 1.46395f)},
    //铑 Rhodium
    {"Rh", vec3(2.58031f, 1.85624f, 1.55114f)},
    {"Rh_palik", vec3(2.58031f, 1.85624f, 1.55114f)},
    //硒 Selenium
    {"Se", vec3(4.08037f, 2.83343f, 2.81458f)},
    {"Se_palik", vec3(4.08037f, 2.83343f, 2.81458f)},
    //六方碳化硅 Hexagonal silicon carbide
    {"SiC", vec3(3.16562f, 2.52061f, 2.47411f)},
    {"SiC_palik", vec3(3.16562f, 2.52061f, 2.47411f)},
    //碲化锡 Tin telluride
    {"SnTe", vec3(4.51562f, 1.97692f, 1.27897f)},
    {"SnTe_palik", vec3(4.51562f, 1.97692f, 1.27897f)},
    //钽 Tantalum
    {"Ta", vec3(2.05820f, 2.38802f, 2.62250f)},
    {"Ta_palik", vec3(2.05820f, 2.38802f, 2.62250f)},
    //三方碲 Trigonal tellurium
    {"Te", vec3(7.37519f, 4.47257f, 2.63149f)},
    {"Te_palik", vec3(7.37519f, 4.47257f, 2.63149f)},
    //多晶四氟化钍 Polycryst. thorium tetrafluoride
    {"ThF4", vec3(1.82684f, 1.43917f, 1.38471f)},
    {"ThF4_palik", vec3(1.82684f, 1.43917f, 1.38471f)},
    //多晶碳化钛 Polycrystalline titanium carbide
    {"TiC", vec3(3.69261f, 2.83141f, 2.57683f)},
    {"TiC_palik", vec3(3.69261f, 2.83141f, 2.57683f)},
    //氮化钛 Titanium nitride
    {"TiN", vec3(1.64497f, 1.14800f, 1.37685f)},
    {"TiN_palik", vec3(1.64497f, 1.14800f, 1.37685f)},
    //四方二氧化钛 Tetragonal titanium dioxide
    {"TiO2", vec3(3.44929f, 2.79576f, 2.89899f)},
    {"TiO2_palik", vec3(3.44929f, 2.79576f, 2.89899f)},
    //碳化钒 Vanadium carbide
    {"VC", vec3(3.64981f, 2.74689f, 2.52731f)},
    {"VC_palik", vec3(3.64981f, 2.74689f, 2.52731f)},
    //钒 Vanadium
    {"V", vec3(4.26843f, 3.50571f, 2.75528f)},
    {"V_palik", vec3(4.26843f, 3.50571f, 2.75528f)},
    //氮化钒 Vanadium nitride
    {"VN", vec3(2.85952f, 2.11468f, 1.93597f)},
    {"VN_palik", vec3(2.85952f, 2.11468f, 1.93597f)},
    //钨 Tungsten
    {"W", vec3(4.36142f, 3.29330f, 2.99191f)},
};

//各种导体材质（conductor）的消光系数（extinction coefficient）
const std::map<std::string, vec3> IOR_k{
    {"a-C", vec3(0.88555f, 0.79763f, 0.81356f)},

    {"Ag", vec3(4.81810f, 3.11562f, 2.14240f)},

    {"Al", vec3(9.20430f, 6.25621f, 4.82675f)},

    {"Al2O3", vec3(2.11325f, 1.66785f, 1.60811f)},

    {"AlAs", vec3(0.00067f, -0.00050f, 0.00741f)},
    {"AlAs_palik", vec3(0.00067f, -0.00050f, 0.00741f)},

    {"AlSb", vec3(-0.02918f, 0.09345f, 1.29784f)},
    {"AlSb_palik", vec3(-0.02918f, 0.09345f, 1.29784f)},

    {"Au", vec3(3.97472f, 2.38066f, 1.59981f)},

    {"Be", vec3(3.82730f, 3.00374f, 2.86293f)},
    {"Be_palik", vec3(3.82730f, 3.00374f, 2.86293f)},

    {"Cr", vec3(5.19538f, 4.22239f, 3.74700f)},

    {"CsI", vec3(0.00000f, 0.00000f, 0.00000f)},
    {"CsI_palik", vec3(0.00000f, 0.00000f, 0.00000f)},

    {"Cu", vec3(3.90464f, 2.44763f, 2.13765f)},
    {"Cu_palik", vec3(3.91324f, 2.45193f, 2.13213f)},

    {"Cu2O", vec3(0.11395f, 0.19341f, 0.60475f)},
    {"Cu2O_palik", vec3(0.11395f, 0.19341f, 0.60475f)},

    {"CuO", vec3(0.51999f, 0.56882f, 0.72064f)},
    {"CuO_palik", vec3(0.51999f, 0.56882f, 0.72064f)},

    {"d-C", vec3(0.00000f, 0.00000f, 0.00000f)},
    {"d-C_palik", vec3(0.00000f, 0.00000f, 0.00000f)},

    {"Fe", vec3(3.83077f, 2.73841f, 2.31812f)},

    {"GaN", vec3(0.00020f, -0.00017f, 0.00128f)},

    {"Ge", vec3(0.50903f, 2.19196f, 2.05716f)},

    {"Hg", vec3(6.31420f, 4.36266f, 3.41454f)},
    {"Hg_palik", vec3(6.31420f, 4.36266f, 3.41454f)},

    {"HgTe", vec3(1.62853f, 1.57746f, 1.72592f)},
    {"HgTe_palik", vec3(1.62853f, 1.57746f, 1.72592f)},

    {"Ir", vec3(5.58028f, 4.05855f, 3.26033f)},
    {"Ir_palik", vec3(5.58028f, 4.05855f, 3.26033f)},

    {"K", vec3(2.09975f, 1.34607f, 0.91128f)},
    {"K_palik", vec3(2.09975f, 1.34607f, 0.91128f)},

    {"Li", vec3(3.53305f, 2.30618f, 1.66505f)},
    {"Li_palik", vec3(3.53305f, 2.30618f, 1.66505f)},

    {"MgO", vec3(0.00000f, -0.00000f, 0.00000f)},
    {"MgO_palik", vec3(0.00000f, -0.00000f, 0.00000f)},

    {"Mo", vec3(4.10240f, 3.41361f, 3.14393f)},
    {"Mo_palik", vec3(4.10240f, 3.41361f, 3.14393f)},

    {"Na", vec3(3.17254f, 2.10800f, 1.57575f)},
    {"Na_palik", vec3(3.17254f, 2.10800f, 1.57575f)},

    {"Nb", vec3(3.43408f, 2.73183f, 2.57445f)},
    {"Nb_palik", vec3(3.43408f, 2.73183f, 2.57445f)},

    {"Ni", vec3(4.48929f, 3.04369f, 2.34046f)},
    {"Ni_palik", vec3(4.48929f, 3.04369f, 2.34046f)},

    {"Rh", vec3(6.76790f, 4.69297f, 3.96766f)},
    {"Rh_palik", vec3(6.76790f, 4.69297f, 3.96766f)},

    {"Se", vec3(0.75378f, 0.63705f, 0.52047f)},
    {"Se_palik", vec3(0.75378f, 0.63705f, 0.52047f)},

    {"SiC", vec3(0.00000f, -0.00000f, 0.00001f)},
    {"SiC_palik", vec3(0.00000f, -0.00000f, 0.00001f)},

    {"SnTe", vec3(0.00000f, 0.00000f, 0.00000f)},
    {"SnTe_palik", vec3(0.00000f, 0.00000f, 0.00000f)},

    {"Ta", vec3(2.40293f, 1.73767f, 1.94291f)},
    {"Ta_palik", vec3(2.40293f, 1.73767f, 1.94291f)},

    {"Te", vec3(3.24919f, 3.51992f, 3.28521f)},
    {"Te_palik", vec3(3.24919f, 3.51992f, 3.28521f)},

    {"ThF4", vec3(0.00000f, 0.00000f, 0.00000f)},
    {"ThF4_palik", vec3(0.00000f, 0.00000f, 0.00000f)},

    {"TiC", vec3(3.25876f, 2.34656f, 2.16818f)},
    {"TiC_palik", vec3(3.25876f, 2.34656f, 2.16818f)},

    {"TiN", vec3(3.36132f, 1.93936f, 1.09967f)},
    {"TiN_palik", vec3(3.36132f, 1.93936f, 1.09967f)},

    {"TiO2", vec3(0.00010f, -0.00009f, 0.00063f)},
    {"TiO2_palik", vec3(0.00010f, -0.00009f, 0.00063f)},

    {"VC", vec3(3.06184f, 2.19400f, 1.95902f)},
    {"VC_palik", vec3(3.06184f, 2.19400f, 1.95902f)},

    {"V", vec3(3.48377f, 2.88322f, 3.10511f)},
    {"V_palik", vec3(3.48377f, 2.88322f, 3.10511f)},

    {"VN", vec3(3.02590f, 2.05174f, 1.61287f)},
    {"VN_palik", vec3(3.02590f, 2.05174f, 1.61287f)},

    {"W", vec3(3.49325f, 2.59934f, 2.26838f)},
};
