#include "volpath.hpp"

#include <iostream>

#include "../accelerators/accelerator.hpp"
#include "../core/intersection.hpp"
#include "../emitters/emitter.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"
#include "../media/medium.hpp"

NAMESPACE_BEGIN(raytracer)

VolPathIntegrator::VolPathIntegrator(int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
                                     const std::vector<Emitter *> &emitters, size_t shape_num)
    : Integrator(IntegratorType::kVolPath, max_depth, rr_depth, hide_emitters, accelerator, emitters, shape_num)
{
}

dvec3 VolPathIntegrator::Shade(const dvec3 &eye, const dvec3 &look_dir, Sampler *sampler) const
{
   auto its = Intersection();
   if (!accelerator_ || !accelerator_->Intersect(Ray(eye, look_dir), sampler, &its))
   { //原初光线源于环境
      if (!hide_emitters_)
      {
         auto L = dvec3(0);
         if (envmap_)
         {
            L += envmap_->radiance(eye, -look_dir);
         }
         for (const std::vector<Emitter *> &lights : {point_lights_, directional_lights_})
         {
            for (Emitter *const &emitter : lights)
            {
               L += emitter->radiance(eye, -look_dir);
            }
         }
         return L;
      }
      else
      {
         return dvec3(0);
      }
   }

   dvec3 wo = -look_dir;
   dvec3 L = dvec3(0),
         attenuation = dvec3(1);

   Medium *medium = its.medium(wo);
   bool scattered = false;
   if (medium != nullptr)
   {
      double actual_distance = 0, pdf_scatter = 0;
      auto medium_attenuation = dvec3(0);
      scattered = medium->SampleDistance(its.distance(), &actual_distance, &pdf_scatter, &medium_attenuation, sampler);
      attenuation *= medium_attenuation / pdf_scatter;
      if (scattered)
      { //光线在传播时发生了散射，实际上来源于更近的地方
         its = Intersection(eye + actual_distance * look_dir, medium);
      }
   }
   if (!scattered)
   {
      if (its.IsAbsorbed())
      { //单面材质物体的背面，只吸收而不反射或折射光线
         return dvec3(0);
      }
      else if (its.IsEmitter() && !hide_emitters_)
      { //原初光线源于发光物体
         return attenuation * its.radiance();
      }
   }

   size_t depth = 1;
   while (depth <= max_depth_ && (depth <= rr_depth_ || sampler->Next1D() < pdf_rr_))
   {
      const dvec3 L_direct = SampleAreaLightsDirect(its, wo, sampler) + SampleOtherEmittersDirect(its, wo, sampler);
      L += L_direct * attenuation;

      SamplingRecord rec = its.Sample(wo, sampler);
      if (rec.type == ScatteringType::kNone)
      {
         break;
      }
      else
      {
         attenuation *= (rec.attenuation / rec.pdf);
      }

      Medium *medium = its.medium(-rec.wi);
      auto its_pre = Intersection();
      const Ray ray = {rec.position, -rec.wi};
      bool hit_surface = accelerator_ && accelerator_->Intersect(ray, sampler, &its_pre);
      const double max_distance = hit_surface ? its_pre.distance() : std::numeric_limits<double>::infinity();
      if (medium != nullptr)
      { //当前散射点在参与介质之中
         double actual_distance = 0, pdf_scatter = 0;
         auto medium_attenuation = dvec3(0);
         bool scattered = medium->SampleDistance(max_distance, &actual_distance, &pdf_scatter, &medium_attenuation, sampler);
         attenuation *= medium_attenuation / pdf_scatter;
         if (scattered)
         { //光线在传播时发生了散射，实际上来源于更近的地方
            its_pre = Intersection(rec.position + actual_distance * (-rec.wi), medium);
            if (depth > rr_depth_)
            { //处理俄罗斯轮盘赌算法
               attenuation /= pdf_rr_;
            }
            its = its_pre;
            wo = rec.wi;
            ++depth;
            continue;
         }
      }

      if (hit_surface)
      { //没有散射，光线来自景物表面
         if (its_pre.IsAbsorbed())
         { //没有散射，光线与单面材质的物体交于物体背面而被吸收
            break;
         }
         else if (its_pre.IsEmitter())
         { //没有散射，按 BSDF 采样来自面光源的直接光照
            if (its.IsHarshLobe())
            {
               L += its_pre.radiance() * attenuation;
            }
            else
            {
               double pdf_direct = PdfAreaLight(its_pre, rec.wi),
                      weight_bsdf = MisWeight(rec.pdf, pdf_direct);
               L += weight_bsdf * its_pre.radiance() * attenuation;
            }
            break;
         }
         else
         { //没有散射，光线来自非发光物体的表面
            if (depth > rr_depth_)
            { //处理俄罗斯轮盘赌算法
               attenuation /= pdf_rr_;
            }
         }
      }
      else
      { //没有散射，光线来自环境
         if (envmap_ != nullptr)
         {
            const double pdf_direct = PdfEnvmap(rec.wi),
                         weight_bsdf = MisWeight(rec.pdf, pdf_direct);
            const dvec3 L_env = weight_bsdf * envmap_->radiance(rec.position, rec.wi) * attenuation;
            L += L_env;
         }
         break;
      }
      its = its_pre;
      wo = rec.wi;
      ++depth;
   }
   return L;
}

dvec3 VolPathIntegrator::SampleAreaLightsDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const
{
   if (its_shape.IsHarshLobe() || area_lights_.empty())
   {
      return dvec3(0);
   }

   int index = static_cast<int>(sampler->Next1D() * area_lights_.size());
   if (index == area_lights_.size())
   {
      --index;
   }

   SamplingRecord direct_rec = area_lights_[index]->Sample(its_shape, wo, sampler, accelerator_);
   if (direct_rec.type == ScatteringType::kNone)
   {
      return dvec3(0);
   }

   Medium *medium = its_shape.medium(-direct_rec.wi);
   if (medium != nullptr)
   {

      auto [attenuation, pdf_scatter] = medium->EvalDistance(false, direct_rec.distance);
      if (pdf_scatter <= kEpsilonPdf)
      {
         return dvec3(0);
      }
      direct_rec.radiance *= (attenuation / pdf_scatter);
   }

   SamplingRecord bsdf_rec = its_shape.Eval(direct_rec.wi, wo);
   if (bsdf_rec.type == ScatteringType::kNone)
   {
      return dvec3(0);
   }

   double pdf_direct = direct_rec.pdf * area_light_num_rcp_;
   if (pdf_direct <= kEpsilonPdf)
   {
      return dvec3(0);
   }

   return MisWeight(pdf_direct, bsdf_rec.pdf) * direct_rec.radiance * bsdf_rec.attenuation / pdf_direct;
}

dvec3 VolPathIntegrator::SampleOtherEmittersDirect(const Intersection &its_shape, const dvec3 &wo, Sampler *sampler) const
{
   auto L = dvec3(0);

   if (envmap_ != nullptr)
   {
      SamplingRecord direct_rec = envmap_->Sample(its_shape, wo, sampler, accelerator_);
      Medium *medium = its_shape.medium(-direct_rec.wi);
      if (direct_rec.type != ScatteringType::kNone && medium == nullptr)
      {
         SamplingRecord bsdf_rec = its_shape.Eval(direct_rec.wi, wo);
         if (bsdf_rec.type != ScatteringType::kNone)
         {
            L += MisWeight(direct_rec.pdf, bsdf_rec.pdf) * direct_rec.radiance *
                 bsdf_rec.attenuation / direct_rec.pdf;
         }
      }
   }
   for (Emitter *const &emitter : point_lights_)
   {
      SamplingRecord direct_rec = emitter->Sample(its_shape, wo, sampler, accelerator_);
      if (direct_rec.type == ScatteringType::kNone)
      {
         continue;
      }
      Medium *medium = its_shape.medium(-direct_rec.wi);
      if (medium != nullptr)
      {

         auto [attenuation, pdf_scatter] = medium->EvalDistance(false, direct_rec.distance);
         if (pdf_scatter <= kEpsilonPdf)
         {
            continue;
         }
         direct_rec.radiance *= (attenuation / pdf_scatter);
      }

      SamplingRecord bsdf_rec = its_shape.Eval(direct_rec.wi, wo);
      if (bsdf_rec.type == ScatteringType::kNone)
      {
         continue;
      }

      L += direct_rec.radiance * bsdf_rec.attenuation;
   }

   for (Emitter *const &emitter : directional_lights_)
   {
      SamplingRecord direct_rec = emitter->Sample(its_shape, wo, sampler, accelerator_);
      if (direct_rec.type == ScatteringType::kNone)
      {
         continue;
      }
      Medium *medium = its_shape.medium(-direct_rec.wi);
      if (medium != nullptr)
      {
         continue;
      }

      SamplingRecord bsdf_rec = its_shape.Eval(direct_rec.wi, wo);
      if (bsdf_rec.type == ScatteringType::kNone)
      {
         continue;
      }

      L += direct_rec.radiance * bsdf_rec.attenuation;
   }

   return L;
}

NAMESPACE_END(raytracer)