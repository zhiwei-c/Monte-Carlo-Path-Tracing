#include "bdpt.hpp"

#include <iostream>

#include "../accelerators/accelerator.hpp"
#include "../emitters/emitter.hpp"
#include "../math/sample.hpp"
#include "../math/sampler.hpp"

NAMESPACE_BEGIN(raytracer)

PathVertex::PathVertex()
    : its(Intersection()),
      rec(SamplingRecord())
{
}

PathVertex::PathVertex(const Intersection &its, const SamplingRecord &rec)
    : its(its),
      rec(rec)
{
}

BdptIntegrator::BdptIntegrator(int max_depth, int rr_depth, bool hide_emitters, Accelerator *accelerator,
                               const std::vector<Emitter *> &emitters, size_t shape_num)
    : Integrator(IntegratorType::kBdpt, max_depth, rr_depth, hide_emitters, accelerator, emitters, shape_num)
{
}

dvec3 BdptIntegrator::Shade(const dvec3 &eye, const dvec3 &look_dir, Sampler *sampler) const
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
   else if (its.IsAbsorbed())
   { //单面材质物体的背面，只吸收而不反射或折射光线
      return dvec3(0);
   }
   else if (its.IsEmitter() && !hide_emitters_)
   { //原初光线源于发光物体
      return its.radiance();
   }

   std::vector<PathVertex> emitter_path = CreateEmitterPath(sampler);
   const dvec3 L = ShadeIterately(its, -look_dir, emitter_path, sampler);
   return L;
}

std::vector<PathVertex> BdptIntegrator::CreateEmitterPath(Sampler *sampler) const
{
   auto path = std::vector<PathVertex>();
   if (area_lights_.empty())
   {
      return path;
   }

   int light_index = static_cast<int>(sampler->Next1D() * area_lights_.size());
   if (light_index == area_lights_.size())
   {
      --light_index;
   }
   Intersection its_emitter = area_lights_[light_index]->SamplePoint(sampler);
   auto wo_emitter_inv = dvec3(0);
   double pdf = SampleHemisCos(its_emitter.normal(), &wo_emitter_inv, sampler->Next2D());
   SamplingRecord rec_emitter = its_emitter.PackGeomtryInfo(-wo_emitter_inv);
   rec_emitter.pdf = pdf;
   rec_emitter.wo = -wo_emitter_inv;
   path.emplace_back(its_emitter, rec_emitter);

   size_t depth = 0;
   while (depth < max_depth_ && (depth < rr_depth_ || sampler->Next1D() < pdf_rr_))
   {
      const PathVertex prev = path.back();
      const dvec3 position_prev = prev.its.position(),
                  wo_prev = prev.rec.wo;

      const Ray ray = {position_prev, wo_prev};
      auto its_now = Intersection();
      if (!accelerator_->Intersect(ray, sampler, &its_now) || its_now.IsAbsorbed() || its_now.IsEmitter())
      {
         break;
      }
      const dvec3 wi_now = wo_prev;
      const SamplingRecord rec_now_pseudo = its_now.Sample(-wi_now, sampler);
      const SamplingRecord rec_now = its_now.Eval(wi_now, -rec_now_pseudo.wi);
      if (rec_now.type == ScatteringType::kNone)
      {
         break;
      }
      path.emplace_back(its_now, rec_now);
      ++depth;
   }

   if (path.size() > 1)
   {
      auto L_area_bsdf = dvec3(0);
      double cos_theta_prime = glm::dot(path[0].rec.wo, path[0].rec.normal),
             pdf_area = path[0].its.pdf_area() * area_light_num_rcp_,
             distance = glm::length(path[0].rec.position - path[1].rec.position),
             pdf_direct = pdf_area * distance * distance / cos_theta_prime;
      if (pdf_direct > kEpsilonPdf)
      {
         double weight_bsdf = MisWeight(path[1].rec.pdf, pdf_direct);
         L_area_bsdf = weight_bsdf * path[0].rec.radiance * (path[1].rec.attenuation / path[1].rec.pdf);
      }
      const dvec3 L_light_direct = SampleAreaLightsDirect(path[1].its, path[1].rec.wo, sampler) +
                                   SampleOtherEmittersDirect(path[1].its, path[1].rec.wo, sampler);
      path[1].rec.radiance = L_area_bsdf + L_light_direct;

      for (size_t i = 2; i < path.size() - 1; ++i)
      {
         const dvec3 L_direct = SampleAreaLightsDirect(path[i].its, path[i].rec.wo, sampler) +
                                SampleOtherEmittersDirect(path[i].its, path[i].rec.wo, sampler);
         auto L_indirect = dvec3(0);
         double cos_theta_prime = std::abs(glm::dot(path[i - 1].rec.wo, path[i - 1].rec.normal)),
                pdf_area = path[i - 1].its.pdf_area() * no_emitter_num_rcp_,
                distance = glm::length(path[i].rec.position - path[i - 1].rec.position),
                pdf_direct = pdf_area * distance * distance / cos_theta_prime;
         if (pdf_direct > kEpsilonPdf)
         {
            double weight_direct = MisWeight(pdf_direct, path[i].rec.pdf);
            L_indirect = weight_direct * path[i - 1].rec.radiance * (path[i].rec.attenuation / pdf_direct);
         }

         path[i].rec.radiance = L_direct + L_indirect;
         if (i > pdf_rr_)
         {
            path[i].rec.radiance /= pdf_rr_;
         }
      }
   }

   return path;
}

dvec3 BdptIntegrator::ShadeIterately(Intersection its, dvec3 wo, const std::vector<PathVertex> &emitter_path, Sampler *sampler) const
{
   dvec3 L = dvec3(0),
         attenuation = dvec3(1);
   size_t depth = 1;
   while (depth <= max_depth_ && (depth <= rr_depth_ || sampler->Next1D() < pdf_rr_))
   {
      const dvec3 L_direct = SampleAreaLightsDirect(its, wo, sampler) + SampleOtherEmittersDirect(its, wo, sampler);
      L += L_direct * attenuation;

      if (!its.IsHarshLobe())
      {
         std::vector<dvec3> L_indirects;
         std::vector<double> pdf_indirects;
         for (size_t i = 1; i < emitter_path.size(); ++i)
         {
            const PathVertex &prev = emitter_path[i];
            const dvec3 wo_prev = glm::normalize(its.position() - prev.rec.position);
            double cos_theta_prime = std::abs(glm::dot(wo_prev, prev.rec.normal));
            if (cos_theta_prime == 0.0)
            {
               continue;
            }

            if (!Visible(its, prev.its, sampler))
            {
               continue;
            }

            SamplingRecord rec_prev = prev.its.Eval(prev.rec.wi, wo_prev);
            if (rec_prev.type == ScatteringType::kNone)
            {
               continue;
            }

            const dvec3 wi_now = wo_prev;
            SamplingRecord rec_now = its.Eval(wi_now, wo);
            if (rec_now.type == ScatteringType::kNone)
            {
               continue;
            }

            const PathVertex &prev_prev = emitter_path[i - 1];

            auto L_prev = dvec3(0);
            if (i == 1)
            {
               L_prev += SampleOtherEmittersDirect(prev.its, wo_prev, sampler);
               double pdf_area = prev_prev.its.pdf_area() * area_light_num_rcp_,
                      distance = glm::length(prev.rec.position - prev_prev.rec.position),
                      pdf_direct = pdf_area * distance * distance / cos_theta_prime;
               if (pdf_direct > kEpsilonPdf)
               {
                  double weight_direct = MisWeight(pdf_direct, rec_prev.pdf);
                  L_prev += weight_direct * prev_prev.rec.radiance * rec_prev.attenuation / pdf_direct;
               }
            }
            else
            {
               auto L_indirect_prev = dvec3(0);
               double pdf_area = prev_prev.its.pdf_area() * no_emitter_num_rcp_,
                      distance = glm::length(prev.rec.position - prev_prev.rec.position),
                      pdf_direct = pdf_area * distance * distance / cos_theta_prime;
               if (pdf_direct > kEpsilonPdf)
               {
                  double weight_direct = MisWeight(pdf_direct, rec_prev.pdf);
                  L_indirect_prev = weight_direct * prev_prev.rec.radiance * rec_prev.attenuation / pdf_direct;
               }

               const dvec3 L_direct_prev = SampleAreaLightsDirect(prev.its, wo_prev, sampler) +
                                           SampleOtherEmittersDirect(prev.its, wo_prev, sampler);
               L_prev = L_direct_prev + L_indirect_prev;
            }
            if (i > rr_depth_)
            {
               L_prev /= pdf_rr_;
            }

            double pdf_area = prev.its.pdf_area() * no_emitter_num_rcp_,
                   distance = glm::length(its.position() - prev.rec.position),
                   pdf_direct = pdf_area * distance * distance / cos_theta_prime;
            if (pdf_direct > kEpsilonPdf)
            {
               double weight_direct = MisWeight(pdf_direct, rec_now.pdf);
               L_indirects.push_back(weight_direct * L_prev * rec_now.attenuation / pdf_direct);
               pdf_indirects.push_back(pdf_direct);
            }
         }
         const dvec3 L_indirect_d = WeightPowerHeuristic(L_indirects, pdf_indirects);
         L += L_indirect_d * attenuation;
      }

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
      if (accelerator_->Intersect(ray, sampler, &its_pre))
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
         else
         {
            double cos_theta_prime = std::max(std::abs(glm::dot(rec.wi, its_pre.normal())), kEpsilonCompare),
                   pdf_area = its_pre.pdf_area() * no_emitter_num_rcp_,
                   distance = glm::length(its.position() - its_pre.position()),
                   pdf_direct = pdf_area * distance * distance / cos_theta_prime,
                   weight_bsdf = MisWeight(rec.pdf, pdf_direct);
            attenuation *= weight_bsdf;
            if (depth > rr_depth_)
            {
               attenuation /= pdf_rr_;
            }
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
      wo = rec.wi;
      its = its_pre;
      ++depth;
   }

   return L;
}

NAMESPACE_END(raytracer)