#include "path.h"

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
				return ShadeRecursively(hit, -look_dir, 1);
			}
		}
	}

	if (envmap_ != nullptr)
		return envmap_->GetLe(look_dir);

	return Spectrum(0);
}

Spectrum PathIntegrator::ShadeRecursively(const Intersection &its, const Vector3 &wo, int depth) const
{
	if (max_depth_ > 0 && depth == max_depth_)
		return Spectrum(0);

	Spectrum L_env(0), //来自环境的直接光照
		L_emitter(0),  //来自面光源的直接光照
		L_indirect(0); //间接光照

	//按发光物体表面积采样来自面光源的直接光照
	EmitterDirectArea(its, wo, L_emitter);

	//按BSDF采样光照
	BsdfSampling b_rec = its.Sample(wo);
	if (b_rec.pdf > kEpsilonPdf)
	{
		auto cos_theta = std::fabs(glm::dot(b_rec.wi, its.normal()));

		auto its_pre = this->bvh_->Intersect(Ray(its.pos(), -b_rec.wi));
		if (its_pre.valid())
		{
			if (!its_pre.HasEmission()) //按BSDF采样间接光照
			{
				if (UniformFloat() < pdf_rr_)
				{
					auto L_pre = ShadeRecursively(its_pre, b_rec.wi, depth + 1);
					L_indirect = L_pre * b_rec.weight * cos_theta / (b_rec.pdf * pdf_rr_);
				}
			}
			else //按BSDF采样来自面光源的直接光照，并和按发光物体表面积采样的来自面光源的直接光照根据多重重要性采样技术合并
			{
				auto pdf_direct = PdfEmitterDirect(its_pre, b_rec.wi);
				auto weight_bsdf = MisWeight(b_rec.pdf, pdf_direct);
				L_emitter += weight_bsdf * its_pre.radiance() * b_rec.weight * cos_theta / b_rec.pdf;
			}
		}
		else if (envmap_ != nullptr) //按BSDF采样来自环境的直接光照
		{
			L_env = envmap_->GetLe(-b_rec.wi) * b_rec.weight * cos_theta / b_rec.pdf;
		}
	}

	return L_emitter + L_env + L_indirect;
}

NAMESPACE_END(simple_renderer)