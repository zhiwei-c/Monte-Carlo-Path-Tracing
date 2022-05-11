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
	std::map<std::string, Material *> id_to_material_;
	int bsdf_cnt_;
	Float gamma_;

	void Reset()
	{
		xml_directory_ = "";
		id_to_material_.clear();
		bsdf_cnt_ = 0;
	}

	Integrator *ParseIntegrator(rapidxml::xml_node<> *node_integrator);
	Camera *ParseCamera(rapidxml::xml_node<> *node_sensor);
	void ParseShape(rapidxml::xml_node<> *node_shape, Renderer *renderer);
	void ParseMaterial(rapidxml::xml_node<> *node_bsdf, Renderer *renderer, const std::string &id_default = "");
	Envmap *ParseEnvmap(rapidxml::xml_node<> *node_envmap);
	Film ParseFilm(rapidxml::xml_node<> *node_sensor);

	std::unique_ptr<Texture> ParseBumpMapping(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);
	std::unique_ptr<Texture> ParseOpacity(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);
	bool ParseCoating(const std::string &id, rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type);

	Material *ParseDiffuse(rapidxml::xml_node<> *node_diffuse);
	Material *ParseRoughDiffuse(rapidxml::xml_node<> *node_rough_diffuse);
	Material *ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin = false);
	Material *ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric);
	Material *ParseConductor(rapidxml::xml_node<> *node_conductor);
	Material *ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor);
	Material *ParsePlastic(rapidxml::xml_node<> *node_plastic);
	Material *ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic);

	std::unique_ptr<Texture> ParseTexture(rapidxml::xml_node<> *node_texture);
	std::unique_ptr<Texture> ParseTextureOrOther(rapidxml::xml_node<> *node_parent, std::string name);

	static std::unique_ptr<Mat4> GetToWorld(rapidxml::xml_node<> *node_parent);
	static Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_material_name);
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