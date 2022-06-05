#pragma once

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <optional>
#include <tuple>

#include "rapidxml/rapidxml.hpp"

#include "../../renderer.h"

NAMESPACE_BEGIN(raytracer)

class XmlParser
{
public:
	XmlParser() : bsdf_cnt_(0), gamma_(-1) {}

	Renderer *Parse(const std::string &path);

private:
	std::string xml_directory_;
	std::map<std::string, Bsdf *> id_to_bsdf_;
	std::map<std::string, Medium *> id_to_medium_;
	int bsdf_cnt_;
	int media_cnt_;
	Float gamma_;

	void Reset()
	{
		xml_directory_ = "";
		id_to_bsdf_.clear();
		id_to_medium_.clear();
		bsdf_cnt_ = 0;
		media_cnt_ = 0;
	}

	Integrator *ParseIntegrator(rapidxml::xml_node<> *node_integrator);
	Camera *ParseCamera(rapidxml::xml_node<> *node_sensor);
	void ParseShape(rapidxml::xml_node<> *node_shape, Renderer *renderer);
	void ParseBsdf(rapidxml::xml_node<> *node_bsdf, Renderer *renderer, const std::string &id_default = "");
	bool ParseMedium(rapidxml::xml_node<> *node_medium, Renderer *renderer, const std::string &id_default = "");
	Envmap *ParseEnvmap(rapidxml::xml_node<> *node_envmap);
	Film ParseFilm(rapidxml::xml_node<> *node_sensor);

	std::unique_ptr<PhaseFunction> ParsePhaseFunction(rapidxml::xml_node<> *&node_medium);

	std::unique_ptr<Texture> ParseBumpMapping(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);
	std::unique_ptr<Texture> ParseOpacity(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);
	bool ParseCoating(const std::string &id, rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);

	Bsdf *ParseDiffuse(rapidxml::xml_node<> *node_diffuse);
	Bsdf *ParseRoughDiffuse(rapidxml::xml_node<> *node_rough_diffuse);
	Bsdf *ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin = false);
	Bsdf *ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric);
	Bsdf *ParseConductor(rapidxml::xml_node<> *node_conductor);
	Bsdf *ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor);
	Bsdf *ParsePlastic(rapidxml::xml_node<> *node_plastic);
	Bsdf *ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic);

	std::unique_ptr<Texture> ParseTexture(rapidxml::xml_node<> *node_texture);
	std::unique_ptr<Texture> ParseTextureOrOther(rapidxml::xml_node<> *node_parent, std::string name);

	static std::unique_ptr<Mat4> GetToWorld(rapidxml::xml_node<> *node_parent);
	static Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_bsdf_name);
	static MicrofacetDistribType GetDistrbType(const std::string &name);

	static Spectrum GetSpectrum(rapidxml::xml_node<> *node_spectrum);
	static std::optional<std::string> GetString(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);
	static std::optional<bool> GetBoolean(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);
	static std::optional<int> GetInt(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);
	static std::optional<Float> GetFloat(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);
	static std::optional<Vector3> GetPoint(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

	static std::optional<std::string> GetAttri(rapidxml::xml_node<> *node, std::string key, bool not_exist_ok = false);
	static rapidxml::xml_node<> *GetChild(rapidxml::xml_node<> *node, std::string name, bool not_exist_ok = true);
	static std::string GetTreeName(rapidxml::xml_node<> *node);
};

NAMESPACE_END(raytracer)