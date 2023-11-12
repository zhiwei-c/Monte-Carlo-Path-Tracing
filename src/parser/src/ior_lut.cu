#include "ior_lut.cuh"

#include <vector>

namespace
{

using namespace csrt;

struct DielectricIorInfo
{
    std::string name;
    float ior;
};

struct ConductorIorInfo
{
    std::string name;
    Vec3 eta;
    Vec3 k;
};

const std::vector<DielectricIorInfo> ior_buffer = {
    // 真空
    DielectricIorInfo{"vacuum", 1.0},
    // 溴
    DielectricIorInfo{"bromine", 1.661},
    // 氦
    DielectricIorInfo{"helium", 1.000036},
    // 水冰
    DielectricIorInfo{"water ice", 1.31},
    DielectricIorInfo{"water-ice", 1.31},
    // 氢
    DielectricIorInfo{"hydrogen", 1.000132},
    // 熔融石英
    DielectricIorInfo{"fused quartz", 1.458},
    DielectricIorInfo{"fused-quartz", 1.458},
    // 空气
    DielectricIorInfo{"air", 1.000277},
    // 派热克斯玻璃（一种耐热玻璃）
    DielectricIorInfo{"pyrex", 1.470},
    // 二氧化碳
    DielectricIorInfo{"carbon dioxide", 1.00045},
    DielectricIorInfo{"carbon-dioxide", 1.00045},
    // 丙烯酸玻璃
    DielectricIorInfo{"acrylic glass", 1.49},
    DielectricIorInfo{"acrylic-glass", 1.49},
    // 水
    DielectricIorInfo{"water", 1.3330},
    // 聚丙烯
    DielectricIorInfo{"polypropylene", 1.49},
    // 丙酮
    DielectricIorInfo{"acetone", 1.36},
    // BK7玻璃
    DielectricIorInfo{"bk7", 1.5046},
    // 乙醇
    DielectricIorInfo{"ethanol", 1.361},
    // 氯化钠
    DielectricIorInfo{"sodium chloride", 1.544},
    DielectricIorInfo{"sodium-chloride", 1.544},
    // 四氯化碳
    DielectricIorInfo{"carbon tetrachloride", 1.461},
    DielectricIorInfo{"carbon-tetrachloride", 1.461},
    // 琥珀
    DielectricIorInfo{"amber", 1.55},
    // 甘油
    DielectricIorInfo{"glycerol", 1.4729},
    // 聚对苯二甲酸乙二醇酯
    DielectricIorInfo{"pet", 1.5750},
    // 苯
    DielectricIorInfo{"benzene", 1.501},
    // 金刚石
    DielectricIorInfo{"diamond", 2.419},
    // 硅油
    DielectricIorInfo{"silicone oil", 1.52045},
    DielectricIorInfo{"silicone-oil", 1.52045},
};

const std::vector<ConductorIorInfo> eta_buffer = {
    // 无定形碳 Amorphous carbon
    ConductorIorInfo{"a-C", Vec3(2.93785, 2.22242, 1.96400),
                     Vec3(0.88555, 0.79763, 0.81356)},
    // 银 Silver
    ConductorIorInfo{"Ag", Vec3(0.15494, 0.11648, 0.13809),
                     Vec3(4.81810, 3.11562, 2.14240)},

    // 铝 Aluminium
    ConductorIorInfo{"Al", Vec3(1.65394, 0.87850, 0.52012),
                     Vec3(9.20430, 6.25621, 4.82675)},

    // 氧化铝 Aluminium oxide
    ConductorIorInfo{"Al2O3", Vec3(2.12296, 1.67556, 1.61569),
                     Vec3(2.11325, 1.66785, 1.60811)},

    // 立方砷化铝 Cubic aluminium arsenide
    ConductorIorInfo{"AlAs", Vec3(3.59737, 3.22615, 2.21235),
                     Vec3(0.00067, -0.00050, 0.00741)},

    // 立方锑化铝 Cubic aluminium antimonide
    ConductorIorInfo{"AlSb", Vec3(4.70830, 4.05196, 4.57419),
                     Vec3(-0.02918, 0.09345, 1.29784)},

    // 金 Gold
    ConductorIorInfo{"Au", Vec3(0.14282, 0.37414, 1.43944),
                     Vec3(3.97472, 2.38066, 1.59981)},

    // 多晶铍 Polycrystalline beryllium
    ConductorIorInfo{"Be", Vec3(4.17618, 3.17830, 2.77819),
                     Vec3(3.82730, 3.00374, 2.86293)},

    // 铬 Chromium
    ConductorIorInfo{"Cr", Vec3(4.36041, 2.91052, 1.65119),
                     Vec3(5.19538, 4.22239, 3.74700)},

    // 立方碘化铯 Cubic caesium iodide
    ConductorIorInfo{"CsI", Vec3(2.14035, 1.69870, 1.65890),
                     Vec3(0.00000, 0.00000, 0.00000)},

    // 铜 Copper
    ConductorIorInfo{"Cu", Vec3(0.19999, 0.92209, 1.09988),
                     Vec3(3.90464, 2.44763, 2.13765)},
    ConductorIorInfo{"Cu_palik", Vec3(0.21221, 0.91805, 1.10007),
                     Vec3(3.91324, 2.45193, 2.13213)},

    // 氧化亚铜 Cuprous oxide
    ConductorIorInfo{"Cu2O", Vec3(3.54426, 2.94365, 2.74619),
                     Vec3(0.11395, 0.19341, 0.60475)},

    // 氧化铜 Cupric oxide
    ConductorIorInfo{"CuO", Vec3(3.24062, 2.44261, 2.20556),
                     Vec3(0.51999, 0.56882, 0.72064)},

    // 立方金刚石 Cubic diamond
    ConductorIorInfo{"d-C", Vec3(2.90008, 2.29322, 2.22275),
                     Vec3(0.00000, 0.00000, 0.00000)},

    // 铁 Iron
    ConductorIorInfo{"Fe", Vec3(2.76404, 1.95417, 1.62766),
                     Vec3(3.83077, 2.73841, 2.31812)},

    // 氮化镓 Gallium nitride
    ConductorIorInfo{"GaN", Vec3(2.86583, 2.29184, 2.29848),
                     Vec3(0.00020, -0.00017, 0.00128)},

    // 锗 Germanium
    ConductorIorInfo{"Ge", Vec3(7.05832, 4.78868, 3.57814),
                     Vec3(0.50903, 2.19196, 2.05716)},

    // 汞 Mercury
    ConductorIorInfo{"Hg", Vec3(2.39384, 1.43697, 0.90762),
                     Vec3(6.31420, 4.36266, 3.41454)},

    // 碲化汞 Mercury telluride
    ConductorIorInfo{"HgTe", Vec3(4.76940, 3.22413, 2.65439),
                     Vec3(1.62853, 1.57746, 1.72592)},

    // 铱 Iridium
    ConductorIorInfo{"Ir", Vec3(3.07986, 2.07777, 1.61446),
                     Vec3(5.58028, 4.05855, 3.26033)},

    // 多晶钾 Polycrystalline potassium
    ConductorIorInfo{"K", Vec3(0.06391, 0.04631, 0.03810),
                     Vec3(2.09975, 1.34607, 0.91128)},

    // 锂 Lithium
    ConductorIorInfo{"Li", Vec3(0.26525, 0.19519, 0.22045),
                     Vec3(3.53305, 2.30618, 1.66505)},

    // 氧化镁 Magnesium oxide
    ConductorIorInfo{"MgO", Vec3(2.08521, 1.64721, 1.59150),
                     Vec3(0.00000, -0.00000, 0.00000)},

    // 钼 Molybdenum
    ConductorIorInfo{"Mo", Vec3(4.47417, 3.51799, 2.77018),
                     Vec3(4.10240, 3.41361, 3.14393)},

    // 钠 Sodium
    ConductorIorInfo{"Na", Vec3(0.06014, 0.05602, 0.06186),
                     Vec3(3.17254, 2.10800, 1.57575)},

    // 铌 Niobium
    ConductorIorInfo{"Nb", Vec3(3.41288, 2.78427, 2.39051),
                     Vec3(3.43408, 2.73183, 2.57445)},

    // 镍 Nickel
    ConductorIorInfo{"Ni", Vec3(2.36225, 1.65983, 1.46395),
                     Vec3(4.48929, 3.04369, 2.34046)},

    // 铑 Rhodium
    ConductorIorInfo{"Rh", Vec3(2.58031, 1.85624, 1.55114),
                     Vec3(6.76790, 4.69297, 3.96766)},

    // 硒 Selenium
    ConductorIorInfo{"Se", Vec3(4.08037, 2.83343, 2.81458),
                     Vec3(0.75378, 0.63705, 0.52047)},

    // 六方碳化硅 Hexagonal silicon carbide
    ConductorIorInfo{"SiC", Vec3(3.16562, 2.52061, 2.47411),
                     Vec3(0.00000, -0.00000, 0.00001)},

    // 碲化锡 Tin telluride
    ConductorIorInfo{"SnTe", Vec3(4.51562, 1.97692, 1.27897),
                     Vec3(0.00000, 0.00000, 0.00000)},

    // 钽 Tantalum
    ConductorIorInfo{"Ta", Vec3(2.05820, 2.38802, 2.62250),
                     Vec3(2.40293, 1.73767, 1.94291)},

    // 三方碲 Trigonal tellurium
    ConductorIorInfo{"Te", Vec3(7.37519, 4.47257, 2.63149),
                     Vec3(3.24919, 3.51992, 3.28521)},

    // 多晶四氟化钍 Polycryst. thorium tetrafluoride
    ConductorIorInfo{"ThF4", Vec3(1.82684, 1.43917, 1.38471),
                     Vec3(0.00000, 0.00000, 0.00000)},

    // 多晶碳化钛 Polycrystalline titanium carbide
    ConductorIorInfo{"TiC", Vec3(3.69261, 2.83141, 2.57683),
                     Vec3(3.25876, 2.34656, 2.16818)},

    // 氮化钛 Titanium nitride
    ConductorIorInfo{"TiN", Vec3(1.64497, 1.14800, 1.37685),
                     Vec3(3.36132, 1.93936, 1.09967)},

    // 四方二氧化钛 Tetragonal titanium dioxide
    ConductorIorInfo{"TiO2", Vec3(3.44929, 2.79576, 2.89899),
                     Vec3(0.00010, -0.00009, 0.00063)},

    // 碳化钒 Vanadium carbide
    ConductorIorInfo{"VC", Vec3(3.64981, 2.74689, 2.52731),
                     Vec3(3.06184, 2.19400, 1.95902)},

    // 钒 Vanadium
    ConductorIorInfo{"V", Vec3(4.26843, 3.50571, 2.75528),
                     Vec3(3.48377, 2.88322, 3.10511)},

    // 氮化钒 Vanadium nitride
    ConductorIorInfo{"VN", Vec3(2.85952, 2.11468, 1.93597),
                     Vec3(3.02590, 2.05174, 1.61287)},

    // 钨 Tungsten
    ConductorIorInfo{"W", Vec3(4.36142, 3.29330, 2.99191),
                     Vec3(3.49325, 2.59934, 2.26838)},

    // 100% 反射镜面
    ConductorIorInfo{"none", Vec3(0, 0, 0), Vec3(1, 1, 1)}};

} // namespace

namespace csrt
{
bool ior_lut::LookupDielectricIor(const std::string &name, float *ior)
{

    for (int i = 0; i < ::ior_buffer.size(); ++i)
    {
        if (name == std::string(::ior_buffer[i].name))
        {
            *ior = ::ior_buffer[i].ior;
            return true;
        }
    }
    return false;
}

bool ior_lut::LookupConductorIor(const std::string &name, Vec3 *eta, Vec3 *k)
{
    for (int i = 0; i < ::eta_buffer.size(); ++i)
    {
        if (name == std::string(::eta_buffer[i].name))
        {
            *eta = ::eta_buffer[i].eta;
            *k = ::eta_buffer[i].k;
            return true;
        }
    }
    return false;
}

} // namespace csrt
