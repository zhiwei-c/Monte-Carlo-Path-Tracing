#include "xml_parser.h"

#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <set>

#include "rapidxml/rapidxml_utils.hpp"
#include "glm/gtx/matrix_query.hpp"
#include "../../modeling/model_loader/obj_parser.h"
#include "../file_path.h"

NAMESPACE_BEGIN(simple_renderer)

static Vector3 GetSpectrum(rapidxml::xml_node<> *node_spectrum);

static std::optional<std::string> GetAttri(rapidxml::xml_node<> *node, std::string key, bool not_exist_ok = false);

static rapidxml::xml_node<> *GetChild(rapidxml::xml_node<> *node, std::string name, bool not_exist_ok = true);

static std::unique_ptr<Mat4> GetToWorld(rapidxml::xml_node<> *node_parent);

static Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_material_name);

static std::string GetTreeName(rapidxml::xml_node<> *node);

static std::optional<std::string> GetString(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static std::optional<bool> GetBoolean(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static std::optional<int> GetInt(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static std::optional<Float> GetFloat(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static std::optional<Vector3> GetPoint(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok = true);

static MicrofacetDistribType GetDistrbType(const std::string &name);

std::tuple<Scene *, Camera *> XmlParser::Parse(const std::string &path)
{
	Reset();
	xml_directory_ = GetDirectory(ConvertBackSlash(path));
	auto file_doc = new rapidxml::file<>(path.c_str());
	auto xml_doc = new rapidxml::xml_document<>();
	xml_doc->parse<0>(file_doc->data());

	auto node_scene = xml_doc->first_node("scene");

	auto camera = ParseCamera(node_scene->first_node("sensor"));
	auto scene = ParseScene(node_scene);

	delete file_doc;
	delete xml_doc;

	return {scene, camera};
}

Scene *XmlParser::ParseScene(rapidxml::xml_node<> *node_scene)
{
	std::cout<<"[info] load bsdfs..."<<std::endl;
	auto node_bsdf = node_scene->first_node("bsdf");
	while (node_bsdf)
	{
		ParseBsdf(node_bsdf);
		node_bsdf = node_bsdf->next_sibling("bsdf");
	}

	std::cout<<"[info] load shapes..."<<std::endl;
	auto node_shape = node_scene->first_node("shape");
	while (node_shape)
	{
		ParseShape(node_shape);
		node_shape = node_shape->next_sibling("shape");
	}

	auto envmap = ParseEnvmap(node_scene->first_node("emitter"));
	return new Scene(shapes_, bsdfs_, envmap);
}

Camera *XmlParser::ParseCamera(rapidxml::xml_node<> *node_sensor)
{
	auto type = GetAttri(node_sensor, "type").value();
	if (type != "perspective")
	{
		std::cerr << "[error] " << GetTreeName(node_sensor) << std::endl
				  << "\tcannot handle sensor except from perspective" << std::endl;
		exit(1);
	}

	auto fov_width = GetFloat(node_sensor, "fov", false).value();
	auto to_world = GetToWorld(node_sensor);

	auto eye_pos = Vector3(0, 0, 0);
	auto up = Vector3(0, 1, 0);
	auto look_dir = Vector3(0, 0, 1);
	if (to_world)
	{
		eye_pos = TransfromPt(*to_world, eye_pos);
		up = TransfromDir(*to_world, up);
		look_dir = TransfromDir(*to_world, look_dir);
	}

	auto film = ParseFilm(node_sensor);
	auto fov_height = fov_width * film.height / film.width;

	auto node_sampler = node_sensor->first_node("sampler");
	auto sample_count = GetInt(node_sampler, "sampleCount", false).value();

	return new Camera(film, eye_pos, eye_pos + look_dir, up, fov_height, sample_count);
}

Film XmlParser::ParseFilm(rapidxml::xml_node<> *node_sensor)
{

	auto node_film = node_sensor->first_node("film");
	auto film_type = GetAttri(node_film, "type").value();

	Film film;
	film.width = GetInt(node_film, "width").value_or(768);
	film.height = GetInt(node_film, "height").value_or(576);
	film.format = GetString(node_film, "fileFormat").value_or("png");
	film.gamma = GetFloat(node_film, "gamma").value_or(gamma_);
	gamma_ = film.gamma;
	return film;
}

void XmlParser::ParseBsdf(rapidxml::xml_node<> *node_bsdf, const std::string *id_default)
{
	auto bsdf_type = GetAttri(node_bsdf, "type").value();

	Texture *bump_map = nullptr;
	if (bsdf_type == "bumpmap")
	{
		auto node_bump = node_bsdf->first_node("texture");
		bump_map = ParseTexture(node_bump);
		bump_map->setGamma(1);

		node_bsdf = node_bsdf->first_node("bsdf");
		if (!node_bsdf)
		{
			std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
					  << "\tnot enough bsdf information" << std::endl;
			exit(1);
		}
		bsdf_type = GetAttri(node_bsdf, "type").value();
	}

	auto id = id_default ? *id_default : GetAttri(node_bsdf, "id").value();

	rapidxml::xml_node<> *node_mask = nullptr;
	if (bsdf_type == "mask")
	{
		node_mask = node_bsdf;
		node_bsdf = node_bsdf->first_node("bsdf");
		if (!node_bsdf)
		{
			std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
					  << "\tnot enough bsdf information" << std::endl;
			exit(1);
		}
		bsdf_type = GetAttri(node_bsdf, "type").value();
	}

	bool twsided = false;
	if (bsdf_type == "twosided")
	{
		twsided = true;
		node_bsdf = node_bsdf->first_node("bsdf");
		if (!node_bsdf)
		{
			std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
					  << "\tnot enough bsdf information" << std::endl;
			exit(1);
		}
		bsdf_type = GetAttri(node_bsdf, "type").value();
	}

	if (bsdf_type == "coating" || bsdf_type == "roughcoating")
	{
		std::cout << "[warning] not support coating bsdf, ignore it." << std::endl;

		if (auto node_ref = node_bsdf->first_node("ref"); node_ref)
		{
			if (node_ref->next_sibling("ref"))
			{
				std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
						  << "\tfind multiple ref" << std::endl;
				exit(1);
			}
			auto ref_id = GetAttri(node_ref, "id").value();
			if (bsdf_map_.find(ref_id) == bsdf_map_.end())
			{
				std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
						  << "\tcannot find existed material with id: " << ref_id << std::endl;
				exit(1);
			}
			else
			{
				bsdf_map_[id] = bsdf_map_[ref_id];
				return;
			}
		}

		node_bsdf = node_bsdf->first_node("bsdf");
		if (!node_bsdf)
		{
			std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
					  << "\tnot enough bsdf information" << std::endl;
			exit(1);
		}
		bsdf_type = GetAttri(node_bsdf, "type").value();
	}

	switch (Hash(bsdf_type.c_str()))
	{
	case "diffuse"_hash:
		ParseDiffuse(node_bsdf, id);
		break;
	case "roughdiffuse"_hash:
		std::cout << "[warning] unsupported rough diffuse bsdf, treat as diffuse." << std::endl;
		ParseDiffuse(node_bsdf, id);
		break;
	case "dielectric"_hash:
		twsided = true;
		ParseDielectric(node_bsdf, id);
		break;
	case "roughdielectric"_hash:
		twsided = true;
		ParseRoughDielectric(node_bsdf, id);
		break;
	case "thindielectric"_hash:
		twsided = true;
		ParseDielectric(node_bsdf, id, true);
		break;
	case "conductor"_hash:
		ParseConductor(node_bsdf, id);
		break;
	case "roughconductor"_hash:
		ParseRoughConductor(node_bsdf, id);
		break;
	case "plastic"_hash:
		ParsePlastic(node_bsdf, id);
		break;
	case "roughplastic"_hash:
		ParseRoughPlastic(node_bsdf, id);
		break;
	default:
		std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
				  << "\tcannot handle bsdf type " << bsdf_type << std::endl;
		exit(1);
	}

	bsdfs_.back()->setTwosided(twsided);
	bsdfs_.back()->setBump(bump_map);

	if (node_mask)
	{
		auto node_opacity = GetChild(node_mask, "opacity", false);
		auto opacity_type = node_opacity->name();
		switch (Hash(opacity_type))
		{
		case "rgb"_hash:
		case "spectrum"_hash:
		{
			auto opacity = GetSpectrum(node_opacity);
			if (!FloatEqual(opacity.r, opacity.g) ||
				!FloatEqual(opacity.r, opacity.b) ||
				!FloatEqual(opacity.g, opacity.b))
			{
				std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
						  << "\tnot support different opacity for different color channel" << std::endl;
				exit(1);
			}
			bsdfs_.back()->setOpacity(opacity);
			break;
		}
		case "texture"_hash:
		{
			auto opacity_map = ParseTexture(node_opacity);
			opacity_map->setGamma(1);
			bsdfs_.back()->setOpacity(opacity_map);
			break;
		}
		default:
			std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
					  << "\tcannot handle mask opacity type except from spectrum or texture" << std::endl;
			exit(1);
		}
	}
	bsdf_map_[id] = bsdfs_.back();
}

void XmlParser::ParseDiffuse(rapidxml::xml_node<> *node_diffuse, std::string id)
{
	auto node_reflectance = GetChild(node_diffuse, "reflectance", false);
	auto reflectance_type = node_reflectance->name();
	switch (Hash(reflectance_type))
	{
	case "rgb"_hash:
	case "spectrum"_hash:
	{
		auto reflectance = GetSpectrum(node_reflectance);
		bsdfs_.push_back(new Diffuse(id, reflectance, nullptr));
		break;
	}
	case "texture"_hash:
	{
		auto diffuse_map = ParseTexture(node_reflectance);
		bsdfs_.push_back(new Diffuse(id, Vector3(0.7), diffuse_map));
		break;
	}
	default:
		std::cerr << "[error] " << GetTreeName(node_reflectance) << std::endl
				  << "\tcannot handle diffuse bsdf info except from spectrum or texture" << std::endl;
		exit(1);
	}
}

void XmlParser::ParseDielectric(rapidxml::xml_node<> *node_dielectric, std::string id, bool thin)
{
	auto int_ior = GetIor(node_dielectric, "intIOR", "bk7");
	auto ext_ior = GetIor(node_dielectric, "extIOR", "air");
	if (thin)
		bsdfs_.push_back(new ThinDielectric(id, ext_ior, int_ior));
	else
		bsdfs_.push_back(new Dielectric(id, ext_ior, int_ior));
}

void XmlParser::ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric, std::string id)
{
	auto distri = GetString(node_rough_dielectric, "distribution").value_or("beckmann");
	auto alpha = GetFloat(node_rough_dielectric, "alpha").value_or(0.1);
	auto alpha_u = GetFloat(node_rough_dielectric, "alphaU").value_or(alpha);
	auto alpha_v = GetFloat(node_rough_dielectric, "alphaV").value_or(alpha);
	auto ior_int = GetIor(node_rough_dielectric, "intIOR", "bk7");
	auto ior_ext = GetIor(node_rough_dielectric, "extIOR", "air");
	bsdfs_.push_back(new RoughDielectric(id, ior_ext, ior_int, GetDistrbType(distri), alpha_u, alpha_v));
}

void XmlParser::ParseConductor(rapidxml::xml_node<> *node_conductor, std::string id)
{
	auto ext_eta = GetIor(node_conductor, "extEta", "air");
	auto node_material = GetChild(node_conductor, "material");
	if (node_material)
	{
		auto material_name = GetAttri(node_material, "value").value();
		if (material_name == "none")
		{
			bsdfs_.push_back(new Conductor(id,
										   true,
										   Vector3(0),
										   Vector3(1),
										   ext_eta));
		}
		else if (IOR_eta.find(material_name) != IOR_eta.end())
		{
			bsdfs_.push_back(new Conductor(id,
										   false,
										   IOR_eta.at(material_name),
										   IOR_k.at(material_name),
										   ext_eta));
		}
		else
		{
			std::cerr << "[error] " << GetTreeName(node_material) << std::endl
					  << " unsupported material :" << material_name << ", "
					  << "use default Conductor material instead." << std::endl;
			exit(1);
		}
	}
	else if (node_conductor->first_node() == nullptr)
		bsdfs_.push_back(new Conductor(id, true, Vector3(0), Vector3(1), ext_eta));
	else
	{
		auto node_eta = GetChild(node_conductor, "eta", false);
		auto eta = GetSpectrum(node_eta);
		auto node_k = GetChild(node_conductor, "k", false);
		auto k = GetSpectrum(node_k);
		bsdfs_.push_back(new Conductor(id, false, eta, k, ext_eta));
	}
}

void XmlParser::ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor, std::string id)
{
	auto distri = GetString(node_rough_conductor, "distribution").value_or("beckmann");

	auto alpha = GetFloat(node_rough_conductor, "alpha").value_or(0.1);
	auto alpha_u = GetFloat(node_rough_conductor, "alphaU").value_or(alpha);
	auto alpha_v = GetFloat(node_rough_conductor, "alphaV").value_or(alpha);

	auto ext_eta = GetIor(node_rough_conductor, "extEta", "air");
	auto node_material = GetChild(node_rough_conductor, "material");

	if (node_material)
	{
		auto material_name = GetAttri(node_material, "value").value();
		if (material_name == "none")
		{
			bsdfs_.push_back(new RoughConductor(id,
												true,
												Vector3(0),
												Vector3(1),
												GetDistrbType(distri),
												alpha_u,
												alpha_v,
												ext_eta));
		}
		else if (IOR_eta.find(material_name) != IOR_eta.end())
		{
			bsdfs_.push_back(new RoughConductor(id,
												false,
												IOR_eta.at(material_name),
												IOR_k.at(material_name),
												GetDistrbType(distri),
												alpha_u,
												alpha_v,
												ext_eta));
		}
		else
		{
			std::cerr << "[error] " << GetTreeName(node_material) << std::endl
					  << "unsupported material :" << material_name << ", "
					  << "use default Conductor material instead." << std::endl;
			exit(1);
		}
	}
	else
	{
		auto node_eta = GetChild(node_rough_conductor, "eta", false);
		auto eta = GetSpectrum(node_eta);
		auto node_k = GetChild(node_rough_conductor, "k", false);
		auto k = GetSpectrum(node_k);
		bsdfs_.push_back(new RoughConductor(id, false, eta, k, GetDistrbType(distri), alpha_u, alpha_v, ext_eta));
	}
}

void XmlParser::ParsePlastic(rapidxml::xml_node<> *node_plastic, std::string id)
{
	auto ior_int = GetIor(node_plastic, "intIOR", "polypropylene");
	auto ior_ext = GetIor(node_plastic, "extIOR", "air");
	bool nonlinear = GetBoolean(node_plastic, "nonlinear").value_or(false);
	auto node_reflectance = GetChild(node_plastic, "diffuseReflectance");
	if (!node_reflectance)
	{
		bsdfs_.push_back(new Plastic(id, Vector3(0.5), nullptr, nonlinear, ior_ext, ior_int));
	}
	else
	{
		auto reflectance_type = std::string(node_reflectance->name());
		switch (Hash(reflectance_type.c_str()))
		{
		case "rgb"_hash:
		case "spectrum"_hash:
		{
			auto diffuse_reflectance = GetSpectrum(node_reflectance);
			bsdfs_.push_back(new Plastic(id, diffuse_reflectance, nullptr, nonlinear, ior_ext, ior_int));
			break;
		}
		case "texture"_hash:
		{
			auto diffuse_map = ParseTexture(node_reflectance);
			bsdfs_.push_back(new Plastic(id, Vector3(0.5), diffuse_map, nonlinear, ior_ext, ior_int));
			break;
		}
		default:
		{
			std::cerr << "[error] " << GetTreeName(node_reflectance) << std::endl
					  << "\tcannot handle plastic bsdf reflectance info type except from rgb or texture" << std::endl;
			exit(1);
		}
		}
	}
}

void XmlParser::ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic, std::string id)
{
	auto distri = GetString(node_rough_plastic, "distribution").value_or("beckmann");
	auto alpha = GetFloat(node_rough_plastic, "alpha").value_or(0.1f);
	auto ior_int = GetIor(node_rough_plastic, "intIOR", "polypropylene");
	auto ior_ext = GetIor(node_rough_plastic, "extIOR", "air");
	auto nonlinear = GetBoolean(node_rough_plastic, "nonlinear").value_or(false);
	auto node_reflectance = GetChild(node_rough_plastic, "diffuseReflectance");
	if (!node_reflectance)
	{
		bsdfs_.push_back(new RoughPlastic(id, Vector3(0.5), nullptr, nonlinear, ior_ext, ior_int, GetDistrbType(distri), alpha));
	}
	else
	{
		auto reflectance_type = std::string(node_reflectance->name());
		switch (Hash(reflectance_type.c_str()))
		{
		case "rgb"_hash:
		case "spectrum"_hash:
		{
			auto diffuse_reflectance = GetSpectrum(node_reflectance);
			bsdfs_.push_back(new RoughPlastic(id, diffuse_reflectance, nullptr, nonlinear, ior_ext, ior_int, GetDistrbType(distri), alpha));
			break;
		}
		case "texture"_hash:
		{
			auto diffuse_map = ParseTexture(node_reflectance);
			bsdfs_.push_back(new RoughPlastic(id, Vector3(0.5), diffuse_map, nonlinear, ior_ext, ior_int, GetDistrbType(distri), alpha));
			break;
		}
		default:
		{
			std::cerr << "[error] " << GetTreeName(node_reflectance) << std::endl
					  << "\tcannot handle plastic bsdf reflectance info type except from rgb or texture" << std::endl;
			exit(1);
		}
		}
	}
}

void XmlParser::ParseShape(rapidxml::xml_node<> *node_shape)
{
	std::string ref;
	if (auto node_emitter = node_shape->first_node("emitter"); node_emitter)
	{
		ref = "emitter_unnamed_" + std::to_string(bsdf_cnt_++);
		if (node_emitter->next_sibling("emitter"))
		{
			std::cerr << "[error] " << GetTreeName(node_emitter) << std::endl
					  << "\tfind multiple emitter info" << std::endl;
			exit(1);
		}
		if (GetAttri(node_emitter, "type").value() != "area")
		{
			std::cerr << "[error] " << GetTreeName(node_emitter) << std::endl
					  << "\tcannot handle shape emitter except from area" << std::endl;
			exit(1);
		}
		auto node_radiance = GetChild(node_emitter, "radiance");
		auto radiance = GetSpectrum(node_radiance);
		bsdfs_.push_back(new AreaLight(ref, radiance));
		bsdf_map_[ref] = bsdfs_.back();
	}
	else if (auto node_ref = node_shape->first_node("ref"); node_ref)
	{
		if (node_ref->next_sibling("ref"))
		{
			std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
					  << "\tfind multiple ref" << std::endl;
			exit(1);
		}
		ref = GetAttri(node_ref, "id").value();
	}
	else
	{
		ref = "unnamed_" + std::to_string(bsdf_cnt_++);
		if (auto node_bsdf = node_shape->first_node("bsdf"); node_bsdf)
			ParseBsdf(node_bsdf, &ref);
		else
		{
			std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
					  << "\tcannot find supported bsdf info" << std::endl;
			exit(1);
		}
	}
	auto to_world = GetToWorld(node_shape);
	auto type = GetAttri(node_shape, "type").value();
	auto flip_normals = GetBoolean(node_shape, "flipNormals").value_or(false);

	if (bsdf_map_.find(ref) == bsdf_map_.end())
	{

		std::cerr << "[error] " << GetTreeName(node_shape) << std::endl
				  << "\tcannot find material with name: \"" << ref << "\"" << std::endl;
		exit(1);
	}

	switch (Hash(type.c_str()))
	{
	case "disk"_hash:
		shapes_.push_back(new Disk(bsdf_map_[ref], std::move(to_world), flip_normals));
		break;
	case "sphere"_hash:
	{
		auto radius = GetFloat(node_shape, "radius").value_or(1);
		auto center = GetPoint(node_shape, "center").value_or(Vector3(0));
		shapes_.push_back(new Sphere(bsdf_map_[ref], center, radius, std::move(to_world), flip_normals));
		break;
	}
	case "cube"_hash:
		shapes_.push_back(new Cube(bsdf_map_[ref], std::move(to_world), flip_normals));
		break;
	case "rectangle"_hash:
		shapes_.push_back(new Rectangle(bsdf_map_[ref], std::move(to_world), flip_normals));
		break;
	case "obj"_hash:
	{
		auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
		auto flip_tex_coords = GetBoolean(node_shape, "flipTexCoords").value_or(true);
		auto model_path = xml_directory_ + GetAttri(GetChild(node_shape, "filename", false), "value").value();
		shapes_.push_back(ObjParser::Parse(model_path, bsdf_map_[ref], std::move(to_world), flip_normals, face_normals, flip_tex_coords));
		break;
	}
	case "ply"_hash:
	{
		auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
		auto model_path = xml_directory_ + GetAttri(GetChild(node_shape, "filename", false), "value").value();
		shapes_.push_back(ObjParser::Parse(model_path, bsdf_map_[ref], std::move(to_world), flip_normals, face_normals));
		break;
	}
	default:
		std::cerr << "[error] " << GetTreeName(node_shape) << std::endl
				  << "\tcannot handle shape type " << std::endl;
		exit(1);
	}
}

Envmap *XmlParser::ParseEnvmap(rapidxml::xml_node<> *node_envmap)
{
	if (!node_envmap)
		return nullptr;
	auto envmap_type = GetAttri(node_envmap, "type").value();
	switch (Hash(envmap_type.c_str()))
	{
	case "envmap"_hash:
	{

		auto to_world = GetToWorld(node_envmap);
		auto node_filename = GetChild(node_envmap, "filename", false);
		auto file_name = xml_directory_ + GetAttri(node_filename, "value").value();
		auto gamma = GetFloat(node_envmap, "gamma").value_or(gamma_);
		return new Envmap(file_name, gamma, std::move(to_world));
		break;
	}
	case "constant"_hash:
	{
		auto node_radiance = GetChild(node_envmap, "radiance", false);
		auto radiance = GetSpectrum(node_radiance);
		return new Envmap(radiance);
		break;
	}

	default:
	{
		auto other_result = ParseEnvmap(node_envmap->next_sibling("emitter"));
		if (!other_result)
		{
			std::cout << "[warning] " << GetTreeName(node_envmap) << std::endl
					  << "\tunsupported emitter type, use default envmap instead." << std::endl;
			return new Envmap();
		}
		else
		{
			std::cout << "[warning] " << GetTreeName(node_envmap) << std::endl
					  << "\tunsupported emitter type, ignore it" << std::endl;
			return other_result;
		}
		break;
	}
	}
}

Texture *XmlParser::ParseTexture(rapidxml::xml_node<> *node_texture)
{
	auto texture_type = GetAttri(node_texture, "type").value();
	switch (Hash(texture_type.c_str()))
	{
	case "bitmap"_hash:
	{
		auto node_filename = GetChild(node_texture, "filename", false);
		auto img_path = xml_directory_ + ConvertBackSlash(GetAttri(node_filename, "value").value());
		auto gamma = GetFloat(node_texture, "gamma").value_or(gamma_);
		return new Bitmap(img_path, gamma);
		break;
	}
	case "checkerboard"_hash:
	{
		auto node_color0 = GetChild(node_texture, "color0");
		auto color0 = node_color0 ? GetSpectrum(node_color0) : Vector3(0.4);

		auto node_color1 = GetChild(node_texture, "color1");
		auto color1 = node_color1 ? GetSpectrum(node_color1) : Vector3(0.2);

		auto u_offset = GetFloat(node_texture, "uoffset", false).value(),
			 v_offset = GetFloat(node_texture, "voffset", false).value(),
			 u_scale = GetFloat(node_texture, "uscale", false).value(),
			 v_scale = GetFloat(node_texture, "vscale", false).value();

		return new Checkerboard(color0, color1, {u_offset, v_offset}, {u_scale, v_scale});
		break;
	}
	case "gridtexture"_hash:
	{
		auto node_color0 = GetChild(node_texture, "color0");
		auto color0 = node_color0 ? GetSpectrum(node_color0) : Vector3(0.4);

		auto node_color1 = GetChild(node_texture, "color1");
		auto color1 = node_color1 ? GetSpectrum(node_color1) : Vector3(0.2);

		auto line_width = GetFloat(node_texture, "lineWidth").value_or(0.01),
			 u_offset = GetFloat(node_texture, "uoffset", false).value(),
			 v_offset = GetFloat(node_texture, "voffset", false).value(),
			 u_scale = GetFloat(node_texture, "uscale", false).value(),
			 v_scale = GetFloat(node_texture, "vscale", false).value();

		return new GridTexture(color0, color1, line_width, {u_offset, v_offset}, {u_scale, v_scale});
		break;
	}

	default:
		std::cerr << "[error] " << GetTreeName(node_texture) << std::endl
				  << "\tcannot handle texture type except from bitmap" << std::endl;
		exit(1);
		break;
	}
}

std::optional<std::string> GetAttri(rapidxml::xml_node<> *node, std::string key, bool not_exist_ok)
{
	auto attri = node->first_attribute(key.c_str());
	if (!attri)
	{
		if (not_exist_ok)
			return std::nullopt;
		std::cerr << "[error] " << GetTreeName(node) << std::endl
				  << "\tcannot find " << key << std::endl;
		exit(1);
	}
	if (attri->next_attribute(key.c_str()))
	{
		std::cerr << "[error] " << GetTreeName(node) << std::endl
				  << "\tfind multiple " << key << std::endl;
		exit(1);
	}
	return attri->value();
}

rapidxml::xml_node<> *GetChild(rapidxml::xml_node<> *node, std::string name, bool not_exist_ok)
{
	for (auto child = node->first_node(); child; child = child->next_sibling())
	{
		auto child_name = GetAttri(child, "name", not_exist_ok);
		if (child_name.has_value() && child_name.value() == name)
			return child;
	}
	if (not_exist_ok)
		return nullptr;
	else
	{
		std::cerr << "[error] " << GetTreeName(node) << std::endl
				  << "\tcannot find child node :" << name << std::endl;
		exit(1);
	}
}

std::unique_ptr<Mat4> GetToWorld(rapidxml::xml_node<> *node_parent)
{
	auto node_toworld = GetChild(node_parent, "toWorld");
	if (!node_toworld)
		return nullptr;

	auto node_matrix = node_toworld->first_node("matrix");
	if (!node_matrix || node_matrix->next_sibling() || node_matrix->previous_sibling())
	{
		std::cerr << "[error] " << GetTreeName(node_matrix) << std::endl
				  << "\tcannot handle transform except from matrix or find multiple matrix" << std::endl;
		exit(1);
	}
	auto matrix_str = GetAttri(node_matrix, "value").value();
	Mat4 result;
	auto in_format_str = kFstr;
	for (int i = 0; i < 15; i++)
	{
		in_format_str += (" " + kFstr);
	}
	auto in_format_c = in_format_str.c_str();

#ifdef WIN32
	sscanf_s(matrix_str.c_str(), in_format_c,
			 &result[0][0], &result[1][0], &result[2][0], &result[3][0],
			 &result[0][1], &result[1][1], &result[2][1], &result[3][1],
			 &result[0][2], &result[1][2], &result[2][2], &result[3][2],
			 &result[0][3], &result[1][3], &result[2][3], &result[3][3]);
#else
	sscanf(matrix_str.c_str(), in_format_c,
		   &result[0][0], &result[1][0], &result[2][0], &result[3][0],
		   &result[0][1], &result[1][1], &result[2][1], &result[3][1],
		   &result[0][2], &result[1][2], &result[2][2], &result[3][2],
		   &result[0][3], &result[1][3], &result[2][3], &result[3][3]);
#endif
	if (glm::isIdentity(result, kEpsilon))
		return nullptr;
	else
		return std::make_unique<Mat4>(result);
}

Vector3 GetSpectrum(rapidxml::xml_node<> *node_spectrum)
{
	if (strcmp(node_spectrum->name(), "rgb") != 0)
	{
		std::cerr << "[error] " << GetTreeName(node_spectrum) << std::endl
				  << "\tcannot hanle  spectrum except from rgb" << std::endl;
		exit(1);
	}
	auto value_str = GetAttri(node_spectrum, "value").value();
	Vector3 result;

	auto in_format_str = kFstr;
	for (int i = 0; i < 2; i++)
	{
		in_format_str += (", " + kFstr);
	}
	auto in_format_c = in_format_str.c_str();

#ifdef WIN32
	sscanf_s(value_str.c_str(), in_format_c, &result[0], &result[1], &result[2]);
#else
	sscanf(value_str.c_str(), in_format_c, &result[0], &result[1], &result[2]);
#endif
	return result;
}

std::optional<bool> GetBoolean(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
	auto node_boolean = GetChild(node_parent, name);
	if (!node_boolean)
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
					  << "\tcannot find child node:" << name << std::endl;
			exit(1);
		}
	}
	auto value = GetAttri(node_boolean, "value");
	if (value == "true")
		return true;
	else
		return false;
}

std::optional<std::string> GetString(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
	auto node_string = GetChild(node_parent, name);
	if (!node_string)
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
					  << "\tcannot find child node :" << name << std::endl;
			exit(1);
		}
	}
	else
		return GetAttri(node_string, "value");
}

std::optional<int> GetInt(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
	auto node_int = GetChild(node_parent, name);
	if (!node_int)
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
					  << "\tcannot find child node: " << name << std::endl;
			exit(1);
		}
	}
	if (strcmp(node_int->name(), "integer") != 0)
	{
		std::cerr << "[error] " << GetTreeName(node_int) << std::endl
				  << "\tthe type of \"" << name << "\" provided is not integer" << std::endl;
		exit(1);
	}

	return std::stoi(GetAttri(node_int, "value").value());
}

std::optional<Float> GetFloat(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
	auto node_float = GetChild(node_parent, name);
	if (!node_float)
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
					  << "\tcannot find child node: " << name << std::endl;
			exit(1);
		}
	}

	if (strcmp(node_float->name(), "float") != 0)
	{
		std::cerr << "[error] " << GetTreeName(node_float) << std::endl
				  << "\tthe type of \"" << name << "\" provided is not float" << std::endl;
		exit(1);
	}

	return std::stof(GetAttri(node_float, "value").value());
}

Float GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_material_name)
{
	auto node_ior = GetChild(node_parent, ior_type);
	if (!node_ior)
		return IOR.at(default_material_name);

	if (strcmp(node_ior->name(), "float") == 0)
		return std::stof(GetAttri(node_ior, "value").value());
	else
	{
		auto int_ior_name = GetAttri(node_ior, "value").value();
		if (IOR.find(int_ior_name) != IOR.end())
			return IOR.at(int_ior_name);
		else
		{
			std::cerr << "[error] " << GetTreeName(node_ior) << std::endl
					  << "\tunsupported ior material " << int_ior_name << std::endl;
			exit(1);
		}
	}
}

std::string GetTreeName(rapidxml::xml_node<> *node)
{
	if (!node || node->name_size() == 0)
		return "root";
	else
	{
		auto result = GetTreeName(node->parent()) + " --> " + node->name();
		if (auto attri_name = node->first_attribute("name"); attri_name)
			result = result + ":" + attri_name->value();
		if (auto attri_type = node->first_attribute("type"); attri_type)
			result = result + ":" + attri_type->value();
		if (auto attri_id = node->first_attribute("id"); attri_id)
			result = result + ":" + attri_id->value();
		return result;
	}
}
std::optional<Vector3> GetPoint(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
{
	auto node_point = GetChild(node_parent, name);
	if (!node_point)
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] " << GetTreeName(node_parent) << std::endl
					  << "\tcannot find child node: " << name << std::endl;
			exit(1);
		}
	}

	if (strcmp(node_point->name(), "point") != 0)
	{
		std::cerr << "[error] " << GetTreeName(node_point) << std::endl
				  << "\tthe type of \"" << name << "\" provided is not point" << std::endl;
		exit(1);
	}

	Vector3 result;

	result.x = static_cast<Float>(std::stod(GetAttri(node_point, "x").value()));
	result.y = static_cast<Float>(std::stod(GetAttri(node_point, "y").value()));
	result.z = static_cast<Float>(std::stod(GetAttri(node_point, "z").value()));

	return result;
}

MicrofacetDistribType GetDistrbType(const std::string &name)
{
	switch (Hash(name.c_str()))
	{
	case "beckmann"_hash:
		return MicrofacetDistribType::kBeckmann;
		break;
	case "ggx"_hash:
		return MicrofacetDistribType::kGgx;
		break;
	case "phong"_hash:
		return MicrofacetDistribType::kPhong;
		break;
	default:
		std::cout << "[warning] unkown microfacet distribution: " << name << ", use Beckmann instead.";
		return MicrofacetDistribType::kBeckmann;
		break;
	}
}

NAMESPACE_END(simple_renderer)