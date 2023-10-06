#include "intersection.hpp"

#include "../bsdfs/bsdf.hpp"
#include "../math/coordinate.hpp"
#include "../math/math.hpp"
#include "../media/medium.hpp"

NAMESPACE_BEGIN(raytracer)

/**
 * @brief 光线与景物的交互点类，
 *      没有交互
 */
Intersection::Intersection()
    : type_(IntersectionType::kNone),
      shape_id_(""),
      inside_(false),
      position_(dvec3(0)),
      normal_(dvec3(0)),
      tangent_(dvec3(0)),
      bitangent_(dvec3(0)),
      texcoord_(dvec2(0)),
      distance_(INFINITY),
      bsdf_(nullptr),
      medium_int_(nullptr),
      medium_ext_(nullptr),
      medium_(nullptr),
      pdf_area_(0)
{
}

/**
 * @brief 光线与介质的交互点类，
 *
 * @param position 交互点位置的世界坐标
 * @param medium 交互点所处的介质
 */
Intersection::Intersection(const dvec3 &position, Medium *medium)
    : type_(IntersectionType::kMediumScattering),
      shape_id_(""),
      inside_(false),
      position_(position),
      normal_(dvec3(0)),
      tangent_(dvec3(0)),
      bitangent_(dvec3(0)),
      texcoord_(dvec2(0)),
      distance_(INFINITY),
      bsdf_(nullptr),
      medium_int_(nullptr),
      medium_ext_(nullptr),
      medium_(medium),
      pdf_area_(0)
{
}

/**
 * @brief 光线与景物的交互点类，
 *      景物表面材质仅在正面有效，而光线相较于景物的背面，于是光线被吸收而没有被散射
 *
 * @param shape_id 交互点类对应景物的ID
 * @param distance 光线起点与景物表面交点间的距离
 */
Intersection::Intersection(const std::string &shape_id, double distance, Medium *medium_int, Medium *medium_ext)
    : type_(IntersectionType::kAbsorbing),
      shape_id_(shape_id),
      inside_(false),
      position_(dvec3(0)),
      normal_(dvec3(0)),
      tangent_(dvec3(0)),
      bitangent_(dvec3(0)),
      texcoord_(dvec2(0)),
      distance_(distance),
      bsdf_(nullptr),
      medium_int_(medium_int),
      medium_ext_(medium_ext),
      medium_(nullptr),
      pdf_area_(0)
{
}

/**
 * @brief 光线与景物的交互点类，
 *      主动按表面积抽样而得到的面光源表面交互点
 *
 * @param shape_id 交互点类对应景物的ID
 * @param position 交点空间坐标
 * @param normal 交点发现方向
 * @param bsdf 景物表面材质
 * @param pdf_area 交点处面元对应的概率
 */
Intersection::Intersection(const std::string &shape_id, const dvec3 &position, const dvec3 &normal, double pdf_area,
                           Bsdf *bsdf, Medium *medium_int, Medium *medium_ext)
    : type_(IntersectionType::kEmitting),
      shape_id_(shape_id),
      inside_(false),
      position_(position),
      normal_(normal),
      tangent_(dvec3(0)),
      bitangent_(dvec3(0)),
      texcoord_(dvec2(0)),
      distance_(INFINITY),
      bsdf_(bsdf),
      medium_int_(medium_int),
      medium_ext_(medium_ext),
      medium_(nullptr),
      pdf_area_(pdf_area)
{
}

/**
 * @brief 光线与景物的交互点类，
 *      景物表面材质两面都有效，光线与景物表面相交而被散射
 *
 * @param shape_id 交互点类对应景物的ID
 * @param inside 光线是否交于景物的背面
 * @param pos 交点空间坐标
 * @param normal 交点发现方向
 * @param texcoord 交点纹理坐标
 * @param distance 光线起点与交点之间的距离
 * @param bsdf 景物表面材质
 * @param pdf_area 交点处面元对应的概率
 */
Intersection::Intersection(const std::string &shape_id, bool inside, const dvec3 &position, const dvec3 &normal,
                           const dvec3 &tangent, const dvec3 &bitangent, const dvec2 &texcoord, double distance,
                           double pdf_area, Bsdf *bsdf, Medium *medium_int, Medium *medium_ext)
    : type_(IntersectionType::kSurfaceScattering),
      shape_id_(shape_id),
      inside_(inside),
      position_(position),
      normal_(normal),
      tangent_(tangent),
      bitangent_(bitangent),
      texcoord_(texcoord),
      distance_(distance),
      bsdf_(bsdf),
      medium_int_(medium_int),
      medium_ext_(medium_ext),
      medium_(nullptr),
      pdf_area_(pdf_area)
{
}

SamplingRecord Intersection::Sample(const dvec3 &wo, Sampler *sampler) const
{
  auto rec = SamplingRecord();
  rec.position = position_;
  rec.wo = wo;
  switch (type_)
  {
  case IntersectionType::kSurfaceScattering:
  {
    bool one_side = glm::dot(wo, normal_) > 0;  //光线与交点法线是否同侧
    rec.inside = one_side ? inside_ : !inside_; //法线方向是否指向介质内侧
    rec.normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线出射方向夹角小于90度
    rec.tangent = one_side ? tangent_ : -tangent_;
    rec.bitangent = one_side ? bitangent_ : -bitangent_;
    rec.texcoord = texcoord_;
    if (bsdf_)
    {
      bsdf_->Sample(&rec, sampler);
    }
    else
    {
      rec.type = ScatteringType::kTransimission;
      rec.wi = wo;
      rec.pdf = 1;
      rec.attenuation = dvec3(1);
    }
    break;
  }
  case IntersectionType::kMediumScattering:
    medium_->SamplePhaseFunction(&rec, sampler);
    break;
  default:
    break;
  }

  return rec;
}

SamplingRecord Intersection::Eval(const dvec3 &wi, const dvec3 &wo) const
{
  auto rec = SamplingRecord();
  rec.position = position_;
  rec.wi = wi;
  rec.wo = wo;
  switch (type_)
  {
  case IntersectionType::kSurfaceScattering:
  {
    bool one_side = glm::dot(wi, normal_) < 0;  //入射光线与法线是否同侧
    rec.inside = one_side ? inside_ : !inside_; //法线方向是否指向介质内侧
    rec.normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线入射方向夹角大于90度
    rec.tangent = one_side ? tangent_ : -tangent_;
    rec.bitangent = one_side ? bitangent_ : -bitangent_;
    rec.texcoord = texcoord_;
    if (bsdf_)
    {
      bsdf_->Eval(&rec);
    }
    else if (SameDirection(wi, wo))
    {
      rec.pdf = 1;
      rec.type = ScatteringType::kTransimission;
      rec.attenuation = dvec3(1);
    }
    break;
  }
  case IntersectionType::kMediumScattering:
    medium_->EvalPhaseFunction(&rec);
    break;
  default:
    break;
  }

  return rec;
}

SamplingRecord Intersection::PackGeomtryInfo(const dvec3 &dir) const
{
  bool one_side = glm::dot(dir, normal_) > 0; //方向与法线是否同侧
  auto rec = SamplingRecord();
  rec.inside = one_side ? inside_ : !inside_; //法线方向是否指向介质内侧
  rec.position = position_;
  rec.normal = one_side ? normal_ : -normal_; //处理法线方向，使其与光线入射方向夹角大于90度
  rec.tangent = one_side ? tangent_ : -tangent_;
  rec.bitangent = one_side ? bitangent_ : -bitangent_;
  rec.texcoord = texcoord_;
  rec.radiance = bsdf_->radiance();
  return rec;
}

Medium *Intersection::medium(const dvec3 &dir) const
{
  if (type_ == IntersectionType::kSurfaceScattering)
  {
    bool inside = glm::dot(dir, normal_) > 0.0 ? inside_ : !inside_;
    return inside ? medium_int_ : medium_ext_;
  }
  else
  {
    return medium_;
  }
}

dvec3 Intersection::radiance() const
{
  return bsdf_->radiance();
}

bool Intersection::IsEmitter() const
{
  return bsdf_ != nullptr && bsdf_->IsEmitter();
}

bool Intersection::IsHarshLobe() const
{
  return type_ == IntersectionType::kSurfaceScattering && (bsdf_ == nullptr || bsdf_->IsHarshLobe());
}

NAMESPACE_END(raytracer)