# 蒙特卡洛路径追踪（Monte Carlo Path Tracing）
一个简单的路径追踪小程序，最初参考了《[GAMES101: 现代计算机图形学入门 ](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)》的作业7，有大幅度的调整。

A simple Monte Carlo path tracer based on assignment 7 of [GAMES101]((https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)) originally.

---

绘制示例：

- 《[Country Kitchen](http://www.blendswap.com/blends/view/42851)》 by [Jay-Artist](http://www.blendswap.com/user/Jay-Artist)，来源于 [Rendering Resources | Benedikt Bitterli's Portfolio](https://benedikt-bitterli.me/resources/) 

- 路径追踪算法，16K 分辨率，4 spp 绘制；

![kitchen](resources/rendering_results/kitchen.png)

## 1 依赖（Dependencies）

项目使用 [vcpkg](https://github.com/microsoft/vcpkg) 进行 c++ 库管理。

- [nlohmann json](https://github.com/nlohmann/json)

- [RapidXML](http://rapidxml.sourceforge.net/)

- [glm](https://github.com/g-truc/glm)

- [assimp](https://github.com/assimp/assimp)

- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)

- [stb](https://github.com/nothings/stb)

- [tinyexr](https://github.com/syoyo/tinyexr)

## 2 支持的绘制配置文件格式及使用说明

- 自拟的[ json 格式文件](resources/rendering_resources/cornell-box/scene.json)；
- mitsuba 0.5 定义的[ xml 格式文件](resources/rendering_resources/bathroom/scene.xml)（部分支持）；
- 在 Windows 下编译后执行命令：`.\SimpleRenderer.exe render_config.xml [output_path.png]`

## 3 实现的功能

### 3.1 积分器（Integrators）
- 基于路径追踪（path tracing）算法的积分器，包括：
  - 使用蒙特卡洛方法（Monte Carlo method）计算辐射亮度（radiance）的数学期望；
  - 重要性抽样（importance sampling），给定光线入射方向和表面法线方向，根据 BSDF 对光线出射方向进行重要性抽样；
  - 多重重要性抽样（multiple importance sampling）：
    - 按发光物体表面积直接采样光源；
    - 按 BSDF 采样光源；
  - 俄罗斯轮盘赌（Russian roulette）控制路径追踪深度； 
- 基于双向路径追踪（bidirectional path tracing，BDPT）算法的积分器；

| 绘制参数                       | 路径追踪算法（Path Tracing）                           | 双向路径追踪算法（Bidirectional Path Tracing）         |
| ------------------------------ | ------------------------------------------------------ | ------------------------------------------------------ |
| 640*640 分辨率，<br />4 spp    | ![](resources/rendering_results/box-path-4_spp.png)    | ![](resources/rendering_results/box-bdpt-4_spp.png)    |
| 3840*2160 分辨率，<br />4 spp  | ![](resources/rendering_results/ajar-path-4_spp.png)   | ![](resources/rendering_results/ajar-bdpt-4_spp.png)   |
| 960*960 分辨率，<br />64 spp | ![](resources/rendering_results/bidir-path-64_spp.png) | ![](resources/rendering_results/bidir-bdpt-64_spp.png) |

### 3.2 表面散射模型（Surface scattering models）

- 朗伯模型（Lambert's model）定义的，[平滑的漫反射材质（smooth diffuse material）](src/material/bsdfs/diffuse.h)；

- 冯模型（Phong model）定义的，[有光泽的材质（glossy material）](src/material/bsdfs/glossy.h);

- [平滑的电介质材质（smooth dielectric material）](src/material/bsdfs/dielectric.h)，模仿[ mitsuba 相应的材质](https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#smooth-dielectric-material-dielectric)；

  ![cornell_box-dielectric](resources/rendering_results/cornell_box-dielectric.png)

- 微表面模型（microfacet model）定义的，[粗糙的电介质材质（rough dielectric material）](src/material/bsdfs/rough_dielectric.h)，模仿[ mitsuba 相应的材质](https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#rough-dielectric-material-roughdielectric)；

  ![rough_dielectric](resources/rendering_results/rough_dielectric.png)

- [薄的电介质材质（thin dielectric material）](src/material/bsdfs/thin_dielectric.h)，模仿[ mitsuba 相应的材质](https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#thin-dielectric-material-thindielectric)；

- [平滑的导体材质（smooth conductor material）](src/material/bsdfs/conductor.h)，模仿[ mitsuba 相应的材质](https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#smooth-conductor-conductor)；

  ![Au](resources/rendering_results/Au.png)

- 微表面模型（microfacet model）定义的，[粗糙的导体材质（rough conductor material）](src/material/bsdfs/rough_conductor.h)，模仿[ mitsuba 相应的材质](https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#rough-conductor-material-roughconductor)；

  ![rough_Au](resources/rendering_results/rough_Au.png)

- [平滑的塑料材质（smooth plastic material）](src/material/bsdfs/plastic.h)，模仿[ mitsuba 相应的材质](https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#smooth-plastic-material-plastic)；

  ![plastic](resources/rendering_results/plastic.png)

- [粗糙的塑料材质（rough plastic material）](src/material/bsdfs/rough_plastic.h)，模仿[ mitsuba 相应的材质](https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#rough-plastic-material-roughplastic)；

  ![rough_plastic](resources/rendering_results/rough_plastic.png)



### 3.3 其它

- [使用Kulla和Conty提出的方法](https://fpsunflower.github.io/ckulla/data/s2017_pbs_imageworks_slides_v2.pdf)，尝试补上[微表面模型](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf)没有建模的，微表面之间的多重散射；

- 环境映射（environment mapping）

- 凹凸映射（bump mapping）

- 内置了一些电介质材质的[折射率（refractive index）](src/rendering/ray.h#L9)，在配置文件中可根据名称直接调用：

    | 名称（Preset(s)）                  | 说明（Description）          | 名称（Preset(s)）                              | 说明（Description）  |
    | ---------------------------------- | ---------------------------- | ---------------------------------------------- | -------------------- |
    | vacuum                             | 真空                         | acetone                                        | 丙酮                 |
    | bromine                            | 溴                           | bk7                                            | BK7玻璃              |
    | helium                             | 氦                           | ethanol                                        | 乙醇                 |
    | water ice<br />water-ice           | 水冰                         | sodium chloride<br />sodium-chloride           | 氯化钠               |
    | hydrogen                           | 氢                           | carbon tetrachloride<br />carbon-tetrachloride | 四氯化碳             |
    | fused quartz<br />fused-quartz     | 熔融石英                     | amber                                          | 琥珀                 |
    | air                                | 空气                         | glycerol                                       | 甘油                 |
    | pyrex                              | 派热克斯玻璃（一种耐热玻璃） | pet                                            | 聚对苯二甲酸乙二醇酯 |
    | carbon dioxide<br />carbon-dioxide | 二氧化碳                     | benzene                                        | 苯                   |
    | acrylic glass<br />acrylic-glass   | 丙烯酸玻璃                   | diamond                                        | 金刚石               |
    | water                              | 水                           | silicone oil<br />silicone-oil                 | 硅油                 |
    | polypropylene                      | 聚丙烯                       |                                                |                      |

- 内置了一些导体材质的[折射率](src/rendering/ray.h#L269)和[消光系数（extinction coefficient）](src/rendering/ray.h#L386)，在配置文件中可根据名称直接调用：

    | 名称（Preset(s)）    | 说明（Description）                   | 名称（Preset(s)）    | 说明（Description）                           |
    | -------------------- | ------------------------------------- | -------------------- | --------------------------------------------- |
    | a-C                  | 无定形碳 Amorphous carbon             | Li<br />Li_palik     | 锂 Lithium                                    |
    | Ag                   | 银 Silver                             | MgO<br />MgO_palik   | 氧化镁 Magnesium oxide                        |
    | Al                   | 铝 Aluminium                          | Mo<br />Mo_palik     | 钼 Molybdenum                                 |
    | Al2O3                | 氧化铝 Aluminium oxide                | Na<br />Na_palik     | 钠 Sodium                                     |
    | AlAs<br />AlAs_palik | 立方砷化铝 Cubic aluminium arsenide   | Nb<br />Nb_palik     | 铌 Niobium                                    |
    | AlSb<br />AlSb_palik | 立方锑化铝 Cubic aluminium antimonide | Ni<br />Ni_palik     | 镍 Nickel                                     |
    | Au                   | 金 Gold                               | Rh<br />Rh_palik     | 铑 Rhodium                                    |
    | Be<br />Be_palik     | 多晶铍 Polycrystalline beryllium      | Se<br />Se_palik     | 硒 Selenium                                   |
    | Cr                   | 铬 Chromium                           | SiC<br />SiC_palik   | 六方碳化硅 Hexagonal silicon carbide          |
    | CsI<br />CsI_palik   | 立方碘化铯 Cubic caesium iodide       | SnTe<br />SnTe_palik | 碲化锡 Tin telluride                          |
    | Cu<br />Cu_palik     | 铜 Copper                             | Ta<br />Ta_palik     | 钽 Tantalum                                   |
    | Cu2O<br />Cu2O_palik | 氧化亚铜 Cuprous oxide                | Te<br />Te_palik     | 三方碲 Trigonal tellurium                     |
    | CuO<br />CuO_palik   | 氧化铜 Cupric oxide                   | ThF4<br />ThF4_palik | 多晶四氟化钍 Polycryst. thorium tetrafluoride |
    | d-C<br />d-C_palik   | 立方金刚石 Cubic diamond              | TiC<br />TiC_palik   | 多晶碳化钛 Polycrystalline titanium carbide   |
    | Fe                   | 铁 Iron                               | TiN<br />TiN_palik   | 氮化钛 Titanium nitride                       |
    | GaN                  | 氮化镓 Gallium nitride                | TiO2<br />TiO2_palik | 四方二氧化钛 Tetragonal titanium dioxide      |
    | Ge                   | 锗 Germanium                          | VC<br />VC_palik     | 碳化钒 Vanadium carbide                       |
    | Hg<br />Hg_palik     | 汞 Mercury                            | V<br />V_palik       | 钒 Vanadium                                   |
    | HgTe<br />HgTe_palik | 碲化汞 Mercury telluride              | VN<br />VN_palik     | 氮化钒 Vanadium nitride                       |
    | Ir<br />Ir_palik     | 铱 Iridium                            | W                    | 钨 Tungsten                                   |
    | K<br />K_palik       | 多晶钾 Polycrystalline potassium      | none                 | 全反射镜面 100% reflecting mirror             |

## 4 参考（References）

- 《[GAMES101: 现代计算机图形学入门 ](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html)》

- [Mitsuba renderer](https://github.com/mitsuba-renderer/mitsuba)

- 《[GAMES202: 高质量实时渲染](https://sites.cs.ucsb.edu/~lingqi/teaching/games202.html)》
