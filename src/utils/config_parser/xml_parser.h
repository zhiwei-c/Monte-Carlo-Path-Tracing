#pragma once

#include <string>
#include <vector>
#include <utility>
#include <map>
#include <optional>
#include <tuple>

#include "rapidxml/rapidxml.hpp"

#include "../global.h"
#include "../../rendering/camera.h"
#include "../../modeling/envmap.h"
#include "../../modeling/scene.h"
#include "../../material/texture/textures.h"
#include "../../rendering/integrator.h"

NAMESPACE_BEGIN(simple_renderer)

class XmlParser
{
public:
	XmlParser() : bsdf_cnt_(0), gamma_(-1) {}

	std::tuple<Scene *, Camera *, Integrator *> Parse(const std::string &path);

private:
	std::string xml_directory_;
	std::vector<Material *> bsdfs_;
	std::map<std::string, Material *> bsdf_map_;
	std::vector<Shape *> shapes_;
	int bsdf_cnt_;
	Float gamma_;

	void Reset()
	{
		xml_directory_ = "";
		bsdfs_.clear();
		bsdf_map_.clear();
		shapes_.clear();
		bsdf_cnt_ = 0;
	}
	
	Integrator* ParseIntegrator(rapidxml::xml_node<> *node_integrator);

	Camera *ParseCamera(rapidxml::xml_node<> *node_sensor);

	Scene *ParseScene(rapidxml::xml_node<> *node_scene);

	void ParseShape(rapidxml::xml_node<> *node_shape);

	void ParseBsdf(rapidxml::xml_node<> *node_bsdf, const std::string *id_default = nullptr);

	Envmap *ParseEnvmap(rapidxml::xml_node<> *node_envmap);

	Film ParseFilm(rapidxml::xml_node<> *node_sensor);

	void ParseDiffuse(rapidxml::xml_node<> *node_diffuse, std::string id);

	void ParseDielectric(rapidxml::xml_node<> *node_dielectric, std::string id, bool thin = false);

	void ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric, std::string id);

	void ParseConductor(rapidxml::xml_node<> *node_conductor, std::string id);

	void ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor, std::string id);

	void ParsePlastic(rapidxml::xml_node<> *node_plastic, std::string id);

	void ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic, std::string id);

	Texture *ParseTexture(rapidxml::xml_node<> *node_texture);
};

NAMESPACE_END(simple_renderer)