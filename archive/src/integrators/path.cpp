#include "path.hpp"

#include <iostream>

#include "../accelerators/accelerator.hpp"
#include "../core/intersection.hpp"
#include "../emitters/emitter.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

PathIntegrator::PathIntegrator(int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
                               const std::vector<Emitter *> &emitters, size_t shape_num)
    : Integrator(IntegratorType::kPath, max_depth, rr_depth, hide_emitters, accelerator, emitters, shape_num)
{
}

dvec3 PathIntegrator::Shade(const dvec3 &eye, const dvec3 &look_dir, Sampler *sampler) const
{
   auto its = Intersection();
   if (!accelerator_ || !accelerator_->Intersect(Ray(eye, look_dir), sampler, &its))
   { //原初光线源于环境
      if (!hide_emitters_)
      {
         auto L = dvec3(0);
         if (envmap_ != nullptr)
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
   else if (its.IsAbsorbed())
   { //单面材质物体的背面，只吸收而不反射或折射光线
      return dvec3(0);
   }
   else if (its.IsEmitter() && !hide_emitters_)
   { //原初光线源于发光物体
      return its.radiance();
   }

   dvec3 wo = -look_dir;
   dvec3 L = dvec3(0),
         attenuation = dvec3(1);
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

      auto its_pre = Intersection();
      const Ray ray = {rec.position, -rec.wi};
      if (accelerator_ && accelerator_->Intersect(ray, sampler, &its_pre))
      {
         if (its_pre.IsAbsorbed())
         {
            break;
         }
         else if (its_pre.IsEmitter())
         {
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
         else if (depth > rr_depth_)
         {
            attenuation /= pdf_rr_;
         }
      }
      else
      {
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

NAMESPACE_END(raytracer)