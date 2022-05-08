#pragma once

#include "emitter.h"
#include "kulla_conty.h"

enum MaterialType
{
  kNoneMaterial,    //空材质
  kAreaLight,       //面光源
  kDiffuse,         //漫反射
  kDielectric,      //平滑的电介质
  kRoughDielectric, //粗糙的电介质
  kThinDielectric,  //薄的电介质
  kConductor,       //平滑的导体
  kRoughConductor,  //粗糙的导体
  kPlastic,         //平滑的塑料
  kRoughPlastic,    //粗糙的塑料
};

struct MaterialInfo
{
  MaterialType type;
  bool twosided;
  uint bump_map_idx;
  uint opacity_idx;
  bool mirror;
  vec3 eta;
  vec3 k;
  uint radiance_idx;
  uint diffuse_reflectance_idx;
  uint specular_reflectance_idx;
  uint specular_transmittance_idx;
  MicrofacetDistribType distri;
  uint alpha_u_idx;
  uint alpha_v_idx;
  bool nonlinear;

  __host__ __device__ MaterialInfo()
      : type(kNoneMaterial),
        twosided(false),
        bump_map_idx(kUintMax),
        opacity_idx(kUintMax),
        mirror(true),
        eta(vec3(1)),
        k(vec3(0)),
        radiance_idx(kUintMax),
        diffuse_reflectance_idx(kUintMax),
        specular_reflectance_idx(kUintMax),
        specular_transmittance_idx(kUintMax),
        distri(kNoneDistrib),
        alpha_u_idx(kUintMax),
        alpha_v_idx(kUintMax),
        nonlinear(false) {}
};
