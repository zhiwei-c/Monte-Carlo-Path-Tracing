#pragma once

#include <string>

#include "../utils/global.h"

struct DielectricIorInfo
{
    const char *name;
    Float value;
};

struct ConductorIorInfo
{
    const char *name;
    float3 value;
};

// 各种电介质材质（dielectric）的折射率（refractive index）
constexpr DielectricIorInfo kIor[] = {
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
constexpr ConductorIorInfo kIorEta[] = {
    //无定形碳 Amorphous carbon
    {"a-C", float3{2.93785f, 2.22242f, 1.96400f}},
    //银 Silver
    {"Ag", float3{0.15494f, 0.11648f, 0.13809f}},
    //铝 Aluminium
    {"Al", float3{1.65394f, 0.87850f, 0.52012f}},
    //氧化铝 Aluminium oxide
    {"Al2O3", float3{2.12296f, 1.67556f, 1.61569f}},
    //立方砷化铝 Cubic aluminium arsenide
    {"AlAs", float3{3.59737f, 3.22615f, 2.21235f}},
    {"AlAs_palik", float3{3.59737f, 3.22615f, 2.21235f}},
    //立方锑化铝 Cubic aluminium antimonide
    {"AlSb", float3{4.70830f, 4.05196f, 4.57419f}},
    {"AlSb_palik", float3{4.70830f, 4.05196f, 4.57419f}},
    //金 Gold
    {"Au", float3{0.14282f, 0.37414f, 1.43944f}},
    //多晶铍 Polycrystalline beryllium
    {"Be", float3{4.17618f, 3.17830f, 2.77819f}},
    {"Be_palik", float3{4.17618f, 3.17830f, 2.77819f}},
    //铬 Chromium
    {"Cr", float3{4.36041f, 2.91052f, 1.65119f}},
    //立方碘化铯 Cubic caesium iodide
    {"CsI", float3{2.14035f, 1.69870f, 1.65890f}},
    {"CsI_palik", float3{2.14035f, 1.69870f, 1.65890f}},
    //铜 Copper
    {"Cu", float3{0.19999f, 0.92209f, 1.09988f}},
    {"Cu_palik", float3{0.21221f, 0.91805f, 1.10007f}},
    //氧化亚铜 Cuprous oxide
    {"Cu2O", float3{3.54426f, 2.94365f, 2.74619f}},
    {"Cu2O_palik", float3{3.54426f, 2.94365f, 2.74619f}},
    //氧化铜 Cupric oxide
    {"CuO", float3{3.24062f, 2.44261f, 2.20556f}},
    {"CuO_palik", float3{3.24062f, 2.44261f, 2.20556f}},
    //立方金刚石 Cubic diamond
    {"d-C", float3{2.90008f, 2.29322f, 2.22275f}},
    {"d-C_palik", float3{2.90008f, 2.29322f, 2.22275f}},
    //铁 Iron
    {"Fe", float3{2.76404f, 1.95417f, 1.62766f}},
    //氮化镓 Gallium nitride
    {"GaN", float3{2.86583f, 2.29184f, 2.29848f}},
    //锗 Germanium
    {"Ge", float3{7.05832f, 4.78868f, 3.57814f}},
    //汞 Mercury
    {"Hg", float3{2.39384f, 1.43697f, 0.90762f}},
    {"Hg_palik", float3{2.39384f, 1.43697f, 0.90762f}},
    //碲化汞 Mercury telluride
    {"HgTe", float3{4.76940f, 3.22413f, 2.65439f}},
    {"HgTe_palik", float3{4.76940f, 3.22413f, 2.65439f}},
    //铱 Iridium
    {"Ir", float3{3.07986f, 2.07777f, 1.61446f}},
    {"Ir_palik", float3{3.07986f, 2.07777f, 1.61446f}},
    //多晶钾 Polycrystalline potassium
    {"K", float3{0.06391f, 0.04631f, 0.03810f}},
    {"K_palik", float3{0.06391f, 0.04631f, 0.03810f}},
    //锂 Lithium
    {"Li", float3{0.26525f, 0.19519f, 0.22045f}},
    {"Li_palik", float3{0.26525f, 0.19519f, 0.22045f}},
    //氧化镁 Magnesium oxide
    {"MgO", float3{2.08521f, 1.64721f, 1.59150f}},
    {"MgO_palik", float3{2.08521f, 1.64721f, 1.59150f}},
    //钼 Molybdenum
    {"Mo", float3{4.47417f, 3.51799f, 2.77018f}},
    {"Mo_palik", float3{4.47417f, 3.51799f, 2.77018f}},
    //钠 Sodium
    {"Na", float3{0.06014f, 0.05602f, 0.06186f}},
    {"Na_palik", float3{0.06014f, 0.05602f, 0.06186f}},
    //铌 Niobium
    {"Nb", float3{3.41288f, 2.78427f, 2.39051f}},
    {"Nb_palik", float3{3.41288f, 2.78427f, 2.39051f}},
    //镍 Nickel
    {"Ni", float3{2.36225f, 1.65983f, 1.46395f}},
    {"Ni_palik", float3{2.36225f, 1.65983f, 1.46395f}},
    //铑 Rhodium
    {"Rh", float3{2.58031f, 1.85624f, 1.55114f}},
    {"Rh_palik", float3{2.58031f, 1.85624f, 1.55114f}},
    //硒 Selenium
    {"Se", float3{4.08037f, 2.83343f, 2.81458f}},
    {"Se_palik", float3{4.08037f, 2.83343f, 2.81458f}},
    //六方碳化硅 Hexagonal silicon carbide
    {"SiC", float3{3.16562f, 2.52061f, 2.47411f}},
    {"SiC_palik", float3{3.16562f, 2.52061f, 2.47411f}},
    //碲化锡 Tin telluride
    {"SnTe", float3{4.51562f, 1.97692f, 1.27897f}},
    {"SnTe_palik", float3{4.51562f, 1.97692f, 1.27897f}},
    //钽 Tantalum
    {"Ta", float3{2.05820f, 2.38802f, 2.62250f}},
    {"Ta_palik", float3{2.05820f, 2.38802f, 2.62250f}},
    //三方碲 Trigonal tellurium
    {"Te", float3{7.37519f, 4.47257f, 2.63149f}},
    {"Te_palik", float3{7.37519f, 4.47257f, 2.63149f}},
    //多晶四氟化钍 Polycryst. thorium tetrafluoride
    {"ThF4", float3{1.82684f, 1.43917f, 1.38471f}},
    {"ThF4_palik", float3{1.82684f, 1.43917f, 1.38471f}},
    //多晶碳化钛 Polycrystalline titanium carbide
    {"TiC", float3{3.69261f, 2.83141f, 2.57683f}},
    {"TiC_palik", float3{3.69261f, 2.83141f, 2.57683f}},
    //氮化钛 Titanium nitride
    {"TiN", float3{1.64497f, 1.14800f, 1.37685f}},
    {"TiN_palik", float3{1.64497f, 1.14800f, 1.37685f}},
    //四方二氧化钛 Tetragonal titanium dioxide
    {"TiO2", float3{3.44929f, 2.79576f, 2.89899f}},
    {"TiO2_palik", float3{3.44929f, 2.79576f, 2.89899f}},
    //碳化钒 Vanadium carbide
    {"VC", float3{3.64981f, 2.74689f, 2.52731f}},
    {"VC_palik", float3{3.64981f, 2.74689f, 2.52731f}},
    //钒 Vanadium
    {"V", float3{4.26843f, 3.50571f, 2.75528f}},
    {"V_palik", float3{4.26843f, 3.50571f, 2.75528f}},
    //氮化钒 Vanadium nitride
    {"VN", float3{2.85952f, 2.11468f, 1.93597f}},
    {"VN_palik", float3{2.85952f, 2.11468f, 1.93597f}},
    //钨 Tungsten
    {"W", float3{4.36142f, 3.29330f, 2.99191f}},
    // 100% 反射镜面
    {"none", float3{0, 0, 0}}};

//各种导体材质（conductor）的消光系数（extinction coefficient）
constexpr ConductorIorInfo kIorK[] = {
    {"a-C", float3{0.88555f, 0.79763f, 0.81356f}},

    {"Ag", float3{4.81810f, 3.11562f, 2.14240f}},

    {"Al", float3{9.20430f, 6.25621f, 4.82675f}},

    {"Al2O3", float3{2.11325f, 1.66785f, 1.60811f}},

    {"AlAs", float3{0.00067f, -0.00050f, 0.00741f}},
    {"AlAs_palik", float3{0.00067f, -0.00050f, 0.00741f}},

    {"AlSb", float3{-0.02918f, 0.09345f, 1.29784f}},
    {"AlSb_palik", float3{-0.02918f, 0.09345f, 1.29784f}},

    {"Au", float3{3.97472f, 2.38066f, 1.59981f}},

    {"Be", float3{3.82730f, 3.00374f, 2.86293f}},
    {"Be_palik", float3{3.82730f, 3.00374f, 2.86293f}},

    {"Cr", float3{5.19538f, 4.22239f, 3.74700f}},

    {"CsI", float3{0.00000f, 0.00000f, 0.00000f}},
    {"CsI_palik", float3{0.00000f, 0.00000f, 0.00000f}},

    {"Cu", float3{3.90464f, 2.44763f, 2.13765f}},
    {"Cu_palik", float3{3.91324f, 2.45193f, 2.13213f}},

    {"Cu2O", float3{0.11395f, 0.19341f, 0.60475f}},
    {"Cu2O_palik", float3{0.11395f, 0.19341f, 0.60475f}},

    {"CuO", float3{0.51999f, 0.56882f, 0.72064f}},
    {"CuO_palik", float3{0.51999f, 0.56882f, 0.72064f}},

    {"d-C", float3{0.00000f, 0.00000f, 0.00000f}},
    {"d-C_palik", float3{0.00000f, 0.00000f, 0.00000f}},

    {"Fe", float3{3.83077f, 2.73841f, 2.31812f}},

    {"GaN", float3{0.00020f, -0.00017f, 0.00128f}},

    {"Ge", float3{0.50903f, 2.19196f, 2.05716f}},

    {"Hg", float3{6.31420f, 4.36266f, 3.41454f}},
    {"Hg_palik", float3{6.31420f, 4.36266f, 3.41454f}},

    {"HgTe", float3{1.62853f, 1.57746f, 1.72592f}},
    {"HgTe_palik", float3{1.62853f, 1.57746f, 1.72592f}},

    {"Ir", float3{5.58028f, 4.05855f, 3.26033f}},
    {"Ir_palik", float3{5.58028f, 4.05855f, 3.26033f}},

    {"K", float3{2.09975f, 1.34607f, 0.91128f}},
    {"K_palik", float3{2.09975f, 1.34607f, 0.91128f}},

    {"Li", float3{3.53305f, 2.30618f, 1.66505f}},
    {"Li_palik", float3{3.53305f, 2.30618f, 1.66505f}},

    {"MgO", float3{0.00000f, -0.00000f, 0.00000f}},
    {"MgO_palik", float3{0.00000f, -0.00000f, 0.00000f}},

    {"Mo", float3{4.10240f, 3.41361f, 3.14393f}},
    {"Mo_palik", float3{4.10240f, 3.41361f, 3.14393f}},

    {"Na", float3{3.17254f, 2.10800f, 1.57575f}},
    {"Na_palik", float3{3.17254f, 2.10800f, 1.57575f}},

    {"Nb", float3{3.43408f, 2.73183f, 2.57445f}},
    {"Nb_palik", float3{3.43408f, 2.73183f, 2.57445f}},

    {"Ni", float3{4.48929f, 3.04369f, 2.34046f}},
    {"Ni_palik", float3{4.48929f, 3.04369f, 2.34046f}},

    {"Rh", float3{6.76790f, 4.69297f, 3.96766f}},
    {"Rh_palik", float3{6.76790f, 4.69297f, 3.96766f}},

    {"Se", float3{0.75378f, 0.63705f, 0.52047f}},
    {"Se_palik", float3{0.75378f, 0.63705f, 0.52047f}},

    {"SiC", float3{0.00000f, -0.00000f, 0.00001f}},
    {"SiC_palik", float3{0.00000f, -0.00000f, 0.00001f}},

    {"SnTe", float3{0.00000f, 0.00000f, 0.00000f}},
    {"SnTe_palik", float3{0.00000f, 0.00000f, 0.00000f}},

    {"Ta", float3{2.40293f, 1.73767f, 1.94291f}},
    {"Ta_palik", float3{2.40293f, 1.73767f, 1.94291f}},

    {"Te", float3{3.24919f, 3.51992f, 3.28521f}},
    {"Te_palik", float3{3.24919f, 3.51992f, 3.28521f}},

    {"ThF4", float3{0.00000f, 0.00000f, 0.00000f}},
    {"ThF4_palik", float3{0.00000f, 0.00000f, 0.00000f}},

    {"TiC", float3{3.25876f, 2.34656f, 2.16818f}},
    {"TiC_palik", float3{3.25876f, 2.34656f, 2.16818f}},

    {"TiN", float3{3.36132f, 1.93936f, 1.09967f}},
    {"TiN_palik", float3{3.36132f, 1.93936f, 1.09967f}},

    {"TiO2", float3{0.00010f, -0.00009f, 0.00063f}},
    {"TiO2_palik", float3{0.00010f, -0.00009f, 0.00063f}},

    {"VC", float3{3.06184f, 2.19400f, 1.95902f}},
    {"VC_palik", float3{3.06184f, 2.19400f, 1.95902f}},

    {"V", float3{3.48377f, 2.88322f, 3.10511f}},
    {"V_palik", float3{3.48377f, 2.88322f, 3.10511f}},

    {"VN", float3{3.02590f, 2.05174f, 1.61287f}},
    {"VN_palik", float3{3.02590f, 2.05174f, 1.61287f}},

    {"W", float3{3.49325f, 2.59934f, 2.26838f}},

    {"none", float3{1, 1, 1}}};

constexpr int kDielectricNum = sizeof(kIor) / sizeof(DielectricIorInfo);

constexpr int kConductorNum = sizeof(kIorEta) / sizeof(ConductorIorInfo);

inline bool LookupDielectricIor(const std::string &name, Float &ior)
{
    for (int i = 0; i < kDielectricNum; ++i)
    {
        if (name == std::string(kIor[i].name))
        {
            ior = kIor[i].value;
            return true;
        }
    }
    return false;
}

inline bool LookupConductorIor(const std::string &name, vec3 &eta, vec3 &k)
{
    for (int i = 0; i < kConductorNum; ++i)
    {
        if (name == std::string(kIorEta[i].name))
        {
            eta = vec3(kIorEta[i].value.x, kIorEta[i].value.y, kIorEta[i].value.z);
            k = vec3(kIorK[i].value.x, kIorK[i].value.y, kIorK[i].value.z);
            return true;
        }
    }
    return false;
}
