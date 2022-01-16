#include "path_integrator.h"

NAMESPACE_BEGIN(simple_renderer)

Spectrum PathIntegrator::Shade(const Vector3 &eye_pos, const Vector3 &look_dir) const
{
	if (this->bvh_ != nullptr)
	{
		auto hit = this->bvh_->Intersect(Ray(eye_pos, look_dir));
		if (hit.valid())
		{
			if (hit.HasEmission())
			{
				return hit.radiance();
			}
			else
			{
				return ShadeRecursively(hit, -look_dir);
			}
		}
	}

	if (envmap_ != nullptr)
		return envmap_->GetLe(look_dir);

	return Spectrum(0);
}

Spectrum PathIntegrator::ShadeRecursively(const Intersection &obj, const Vector3 &wo) const
{
	Spectrum L_env(0),  //来自环境的直接光照
		L_emitter(0),  //来自面光源的直接光照
		L_indirect(0); //间接光照

	//按发光物体表面积采样来自面光源的直接光照
	auto [light, pdf_emitter] = SampleEmitter(obj.pos());
	Ray ray_light(obj.pos(), light.pos() - obj.pos());
	auto closet = this->bvh_->Intersect(ray_light);
	if (closet.distance() + kEpsilonDistance > light.distance())
	{
		const auto &wi = -ray_light.dir();
		auto cos_theta_l = glm::dot(wi, light.normal());
		if (cos_theta_l > kEpsilonL)
		{
			auto f_r = obj.Eval(wi, wo);
			if (f_r.r + f_r.g + f_r.b > kEpsilonL)
			{
				auto cos_theta = std::fabs(glm::dot(-wi, obj.normal()));
				const auto &length = light.distance();
				L_emitter = light.radiance() * f_r * static_cast<Float>(cos_theta * cos_theta_l / (Sqr(length) * pdf_emitter));
			}
		}
	}

	//按BSDF采样光照
	BsdfSampling bs = obj.SampleWi(wo);
	if (bs.pdf > kEpsilonL)
	{
		auto cos_theta = std::fabs(glm::dot(-bs.wi, obj.normal()));
		auto pre = this->bvh_->Intersect(Ray(obj.pos(), -bs.wi));
		if (pre.valid())
		{
			if (!pre.HasEmission()) //按BSDF采样间接光照
			{
				if (UniformFloat() < pdf_rr_)
				{
					auto L_pre = ShadeRecursively(pre, bs.wi);
					L_indirect = L_pre * bs.weight * static_cast<Float>(cos_theta / (bs.pdf * pdf_rr_));
				}
			}
			else //按BSDF采样来自面光源的直接光照，并和按发光物体表面积采样的来自面光源的直接光照根据多重重要性采样技术合并
			{
				auto L_emitter_bsdf = pre.radiance() * bs.weight * static_cast<Float>(cos_theta / bs.pdf);
				auto [weight_bsdf, weight_area] = WeightPowerHeuristic(bs.pdf, pdf_emitter);
				L_emitter = weight_bsdf * L_emitter_bsdf +
							weight_area * L_emitter;
			}
		}
		else if (envmap_ != nullptr) //按BSDF采样来自环境的直接光照
		{
			L_env = envmap_->GetLe(-bs.wi) * bs.weight * static_cast<Float>(cos_theta / bs.pdf);
		}
	}

	return L_emitter + L_env + L_indirect;
}

std::pair<Intersection, Float> PathIntegrator::SampleEmitter(const Vector3 &pos_pre) const
{
	Float p = UniformFloat() * this->emit_area_;
	Float emit_area_sum = 0;
	for (size_t k = 0; k < this->emitters_.size(); k++)
	{
		emit_area_sum += this->emitters_[k]->area();
		if (p <= emit_area_sum)
		{
			return this->emitters_[k]->SampleP(pos_pre);
		}
	}
	return {Intersection(), 0};
}

NAMESPACE_END(simple_renderer)