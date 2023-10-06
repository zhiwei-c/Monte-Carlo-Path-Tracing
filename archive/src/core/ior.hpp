#pragma once

#include <array>

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

struct DielectricIorInfo
{
    const char *name;
    double value;
};

struct ConductorIorInfo
{
    const char *name;
    dvec3 value;
};

// 各种电介质材质（dielectric）的折射率（refractive index）
constexpr DielectricIorInfo kIor[] = {
    //真空
    {"vacuum", 1.0},
    //溴
    {"bromine", 1.661},
    //氦
    {"helium", 1.000036},
    //水冰
    {"water ice", 1.31},
    {"water-ice", 1.31},
    //氢
    {"hydrogen", 1.000132},
    //熔融石英
    {"fused quartz", 1.458},
    {"fused-quartz", 1.458},
    //空气
    {"air", 1.000277},
    //派热克斯玻璃（一种耐热玻璃）
    {"pyrex", 1.470},
    //二氧化碳
    {"carbon dioxide", 1.00045},
    {"carbon-dioxide", 1.00045},
    //丙烯酸玻璃
    {"acrylic glass", 1.49},
    {"acrylic-glass", 1.49},
    //水
    {"water", 1.3330},
    //聚丙烯
    {"polypropylene", 1.49},
    //丙酮
    {"acetone", 1.36},
    // BK7玻璃
    {"bk7", 1.5046},
    //乙醇
    {"ethanol", 1.361},
    //氯化钠
    {"sodium chloride", 1.544},
    {"sodium-chloride", 1.544},
    //四氯化碳
    {"carbon tetrachloride", 1.461},
    {"carbon-tetrachloride", 1.461},
    //琥珀
    {"amber", 1.55},
    //甘油
    {"glycerol", 1.4729},
    //聚对苯二甲酸乙二醇酯
    {"pet", 1.5750},
    //苯
    {"benzene", 1.501},
    //金刚石
    {"diamond", 2.419},
    //硅油
    {"silicone oil", 1.52045},
    {"silicone-oil", 1.52045},
};

//各种导体材质（conductor）的折射率（refractive index）
constexpr ConductorIorInfo kIorEta[] = {
    //无定形碳 Amorphous carbon
    {"a-C", dvec3(2.93785, 2.22242, 1.96400)},
    //银 Silver
    {"Ag", dvec3(0.15494, 0.11648, 0.13809)},
    //铝 Aluminium
    {"Al", dvec3(1.65394, 0.87850, 0.52012)},
    //氧化铝 Aluminium oxide
    {"Al2O3", dvec3(2.12296, 1.67556, 1.61569)},
    //立方砷化铝 Cubic aluminium arsenide
    {"AlAs", dvec3(3.59737, 3.22615, 2.21235)},
    {"AlAs_palik", dvec3(3.59737, 3.22615, 2.21235)},
    //立方锑化铝 Cubic aluminium antimonide
    {"AlSb", dvec3(4.70830, 4.05196, 4.57419)},
    {"AlSb_palik", dvec3(4.70830, 4.05196, 4.57419)},
    //金 Gold
    {"Au", dvec3(0.14282, 0.37414, 1.43944)},
    //多晶铍 Polycrystalline beryllium
    {"Be", dvec3(4.17618, 3.17830, 2.77819)},
    {"Be_palik", dvec3(4.17618, 3.17830, 2.77819)},
    //铬 Chromium
    {"Cr", dvec3(4.36041, 2.91052, 1.65119)},
    //立方碘化铯 Cubic caesium iodide
    {"CsI", dvec3(2.14035, 1.69870, 1.65890)},
    {"CsI_palik", dvec3(2.14035, 1.69870, 1.65890)},
    //铜 Copper
    {"Cu", dvec3(0.19999, 0.92209, 1.09988)},
    {"Cu_palik", dvec3(0.21221, 0.91805, 1.10007)},
    //氧化亚铜 Cuprous oxide
    {"Cu2O", dvec3(3.54426, 2.94365, 2.74619)},
    {"Cu2O_palik", dvec3(3.54426, 2.94365, 2.74619)},
    //氧化铜 Cupric oxide
    {"CuO", dvec3(3.24062, 2.44261, 2.20556)},
    {"CuO_palik", dvec3(3.24062, 2.44261, 2.20556)},
    //立方金刚石 Cubic diamond
    {"d-C", dvec3(2.90008, 2.29322, 2.22275)},
    {"d-C_palik", dvec3(2.90008, 2.29322, 2.22275)},
    //铁 Iron
    {"Fe", dvec3(2.76404, 1.95417, 1.62766)},
    //氮化镓 Gallium nitride
    {"GaN", dvec3(2.86583, 2.29184, 2.29848)},
    //锗 Germanium
    {"Ge", dvec3(7.05832, 4.78868, 3.57814)},
    //汞 Mercury
    {"Hg", dvec3(2.39384, 1.43697, 0.90762)},
    {"Hg_palik", dvec3(2.39384, 1.43697, 0.90762)},
    //碲化汞 Mercury telluride
    {"HgTe", dvec3(4.76940, 3.22413, 2.65439)},
    {"HgTe_palik", dvec3(4.76940, 3.22413, 2.65439)},
    //铱 Iridium
    {"Ir", dvec3(3.07986, 2.07777, 1.61446)},
    {"Ir_palik", dvec3(3.07986, 2.07777, 1.61446)},
    //多晶钾 Polycrystalline potassium
    {"K", dvec3(0.06391, 0.04631, 0.03810)},
    {"K_palik", dvec3(0.06391, 0.04631, 0.03810)},
    //锂 Lithium
    {"Li", dvec3(0.26525, 0.19519, 0.22045)},
    {"Li_palik", dvec3(0.26525, 0.19519, 0.22045)},
    //氧化镁 Magnesium oxide
    {"MgO", dvec3(2.08521, 1.64721, 1.59150)},
    {"MgO_palik", dvec3(2.08521, 1.64721, 1.59150)},
    //钼 Molybdenum
    {"Mo", dvec3(4.47417, 3.51799, 2.77018)},
    {"Mo_palik", dvec3(4.47417, 3.51799, 2.77018)},
    //钠 Sodium
    {"Na", dvec3(0.06014, 0.05602, 0.06186)},
    {"Na_palik", dvec3(0.06014, 0.05602, 0.06186)},
    //铌 Niobium
    {"Nb", dvec3(3.41288, 2.78427, 2.39051)},
    {"Nb_palik", dvec3(3.41288, 2.78427, 2.39051)},
    //镍 Nickel
    {"Ni", dvec3(2.36225, 1.65983, 1.46395)},
    {"Ni_palik", dvec3(2.36225, 1.65983, 1.46395)},
    //铑 Rhodium
    {"Rh", dvec3(2.58031, 1.85624, 1.55114)},
    {"Rh_palik", dvec3(2.58031, 1.85624, 1.55114)},
    //硒 Selenium
    {"Se", dvec3(4.08037, 2.83343, 2.81458)},
    {"Se_palik", dvec3(4.08037, 2.83343, 2.81458)},
    //六方碳化硅 Hexagonal silicon carbide
    {"SiC", dvec3(3.16562, 2.52061, 2.47411)},
    {"SiC_palik", dvec3(3.16562, 2.52061, 2.47411)},
    //碲化锡 Tin telluride
    {"SnTe", dvec3(4.51562, 1.97692, 1.27897)},
    {"SnTe_palik", dvec3(4.51562, 1.97692, 1.27897)},
    //钽 Tantalum
    {"Ta", dvec3(2.05820, 2.38802, 2.62250)},
    {"Ta_palik", dvec3(2.05820, 2.38802, 2.62250)},
    //三方碲 Trigonal tellurium
    {"Te", dvec3(7.37519, 4.47257, 2.63149)},
    {"Te_palik", dvec3(7.37519, 4.47257, 2.63149)},
    //多晶四氟化钍 Polycryst. thorium tetrafluoride
    {"ThF4", dvec3(1.82684, 1.43917, 1.38471)},
    {"ThF4_palik", dvec3(1.82684, 1.43917, 1.38471)},
    //多晶碳化钛 Polycrystalline titanium carbide
    {"TiC", dvec3(3.69261, 2.83141, 2.57683)},
    {"TiC_palik", dvec3(3.69261, 2.83141, 2.57683)},
    //氮化钛 Titanium nitride
    {"TiN", dvec3(1.64497, 1.14800, 1.37685)},
    {"TiN_palik", dvec3(1.64497, 1.14800, 1.37685)},
    //四方二氧化钛 Tetragonal titanium dioxide
    {"TiO2", dvec3(3.44929, 2.79576, 2.89899)},
    {"TiO2_palik", dvec3(3.44929, 2.79576, 2.89899)},
    //碳化钒 Vanadium carbide
    {"VC", dvec3(3.64981, 2.74689, 2.52731)},
    {"VC_palik", dvec3(3.64981, 2.74689, 2.52731)},
    //钒 Vanadium
    {"V", dvec3(4.26843, 3.50571, 2.75528)},
    {"V_palik", dvec3(4.26843, 3.50571, 2.75528)},
    //氮化钒 Vanadium nitride
    {"VN", dvec3(2.85952, 2.11468, 1.93597)},
    {"VN_palik", dvec3(2.85952, 2.11468, 1.93597)},
    //钨 Tungsten
    {"W", dvec3(4.36142, 3.29330, 2.99191)},
    // 100% 反射镜面
    {"none", dvec3(0, 0, 0)}};

//各种导体材质（conductor）的消光系数（extinction coefficient）
constexpr ConductorIorInfo kIorK[] = {
    {"a-C", dvec3(0.88555, 0.79763, 0.81356)},

    {"Ag", dvec3(4.81810, 3.11562, 2.14240)},

    {"Al", dvec3(9.20430, 6.25621, 4.82675)},

    {"Al2O3", dvec3(2.11325, 1.66785, 1.60811)},

    {"AlAs", dvec3(0.00067, -0.00050, 0.00741)},
    {"AlAs_palik", dvec3(0.00067, -0.00050, 0.00741)},

    {"AlSb", dvec3(-0.02918, 0.09345, 1.29784)},
    {"AlSb_palik", dvec3(-0.02918, 0.09345, 1.29784)},

    {"Au", dvec3(3.97472, 2.38066, 1.59981)},

    {"Be", dvec3(3.82730, 3.00374, 2.86293)},
    {"Be_palik", dvec3(3.82730, 3.00374, 2.86293)},

    {"Cr", dvec3(5.19538, 4.22239, 3.74700)},

    {"CsI", dvec3(0.00000, 0.00000, 0.00000)},
    {"CsI_palik", dvec3(0.00000, 0.00000, 0.00000)},

    {"Cu", dvec3(3.90464, 2.44763, 2.13765)},
    {"Cu_palik", dvec3(3.91324, 2.45193, 2.13213)},

    {"Cu2O", dvec3(0.11395, 0.19341, 0.60475)},
    {"Cu2O_palik", dvec3(0.11395, 0.19341, 0.60475)},

    {"CuO", dvec3(0.51999, 0.56882, 0.72064)},
    {"CuO_palik", dvec3(0.51999, 0.56882, 0.72064)},

    {"d-C", dvec3(0.00000, 0.00000, 0.00000)},
    {"d-C_palik", dvec3(0.00000, 0.00000, 0.00000)},

    {"Fe", dvec3(3.83077, 2.73841, 2.31812)},

    {"GaN", dvec3(0.00020, -0.00017, 0.00128)},

    {"Ge", dvec3(0.50903, 2.19196, 2.05716)},

    {"Hg", dvec3(6.31420, 4.36266, 3.41454)},
    {"Hg_palik", dvec3(6.31420, 4.36266, 3.41454)},

    {"HgTe", dvec3(1.62853, 1.57746, 1.72592)},
    {"HgTe_palik", dvec3(1.62853, 1.57746, 1.72592)},

    {"Ir", dvec3(5.58028, 4.05855, 3.26033)},
    {"Ir_palik", dvec3(5.58028, 4.05855, 3.26033)},

    {"K", dvec3(2.09975, 1.34607, 0.91128)},
    {"K_palik", dvec3(2.09975, 1.34607, 0.91128)},

    {"Li", dvec3(3.53305, 2.30618, 1.66505)},
    {"Li_palik", dvec3(3.53305, 2.30618, 1.66505)},

    {"MgO", dvec3(0.00000, -0.00000, 0.00000)},
    {"MgO_palik", dvec3(0.00000, -0.00000, 0.00000)},

    {"Mo", dvec3(4.10240, 3.41361, 3.14393)},
    {"Mo_palik", dvec3(4.10240, 3.41361, 3.14393)},

    {"Na", dvec3(3.17254, 2.10800, 1.57575)},
    {"Na_palik", dvec3(3.17254, 2.10800, 1.57575)},

    {"Nb", dvec3(3.43408, 2.73183, 2.57445)},
    {"Nb_palik", dvec3(3.43408, 2.73183, 2.57445)},

    {"Ni", dvec3(4.48929, 3.04369, 2.34046)},
    {"Ni_palik", dvec3(4.48929, 3.04369, 2.34046)},

    {"Rh", dvec3(6.76790, 4.69297, 3.96766)},
    {"Rh_palik", dvec3(6.76790, 4.69297, 3.96766)},

    {"Se", dvec3(0.75378, 0.63705, 0.52047)},
    {"Se_palik", dvec3(0.75378, 0.63705, 0.52047)},

    {"SiC", dvec3(0.00000, -0.00000, 0.00001)},
    {"SiC_palik", dvec3(0.00000, -0.00000, 0.00001)},

    {"SnTe", dvec3(0.00000, 0.00000, 0.00000)},
    {"SnTe_palik", dvec3(0.00000, 0.00000, 0.00000)},

    {"Ta", dvec3(2.40293, 1.73767, 1.94291)},
    {"Ta_palik", dvec3(2.40293, 1.73767, 1.94291)},

    {"Te", dvec3(3.24919, 3.51992, 3.28521)},
    {"Te_palik", dvec3(3.24919, 3.51992, 3.28521)},

    {"ThF4", dvec3(0.00000, 0.00000, 0.00000)},
    {"ThF4_palik", dvec3(0.00000, 0.00000, 0.00000)},

    {"TiC", dvec3(3.25876, 2.34656, 2.16818)},
    {"TiC_palik", dvec3(3.25876, 2.34656, 2.16818)},

    {"TiN", dvec3(3.36132, 1.93936, 1.09967)},
    {"TiN_palik", dvec3(3.36132, 1.93936, 1.09967)},

    {"TiO2", dvec3(0.00010, -0.00009, 0.00063)},
    {"TiO2_palik", dvec3(0.00010, -0.00009, 0.00063)},

    {"VC", dvec3(3.06184, 2.19400, 1.95902)},
    {"VC_palik", dvec3(3.06184, 2.19400, 1.95902)},

    {"V", dvec3(3.48377, 2.88322, 3.10511)},
    {"V_palik", dvec3(3.48377, 2.88322, 3.10511)},

    {"VN", dvec3(3.02590, 2.05174, 1.61287)},
    {"VN_palik", dvec3(3.02590, 2.05174, 1.61287)},

    {"W", dvec3(3.49325, 2.59934, 2.26838)},

    {"none", dvec3(1, 1, 1)}};

constexpr int kDielectricNum = sizeof(kIor) / sizeof(DielectricIorInfo);

constexpr int kConductorNum = sizeof(kIorEta) / sizeof(ConductorIorInfo);

inline bool LookupDielectricIor(const std::string &name, double *ior)
{
    for (int i = 0; i < kDielectricNum; ++i)
    {
        if (name == std::string(kIor[i].name))
        {
            *ior = kIor[i].value;
            return true;
        }
    }
    return false;
}

inline bool LookupConductorIor(const std::string &name, dvec3 *eta, dvec3 *k)
{
    for (int i = 0; i < kConductorNum; ++i)
    {
        if (name == std::string(kIorEta[i].name))
        {
            *eta = kIorEta[i].value;
            *k = kIorK[i].value;
            return true;
        }
    }
    return false;
}

NAMESPACE_END(raytracer)