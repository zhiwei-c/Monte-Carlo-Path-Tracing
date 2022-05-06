#include "xml_parser.h"

#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <set>

#include "rapidxml/rapidxml_utils.hpp"
#include "glm/gtx/matrix_query.hpp"

NAMESPACE_BEGIN(simple_renderer)

Renderer *XmlParser::Parse(const std::string &path)
{
	Reset();
	auto renderer = new Renderer();

	xml_directory_ = GetDirectory(ConvertBackSlash(path));
	auto file_doc = new rapidxml::file<>(path.c_str());
	auto xml_doc = new rapidxml::xml_document<>();
	xml_doc->parse<0>(file_doc->data());
	auto node_scene = xml_doc->first_node("scene");

	std::cout << "[info] load bsdfs..." << std::endl;
	auto node_bsdf = node_scene->first_node("bsdf");
	while (node_bsdf)
	{
		ParseMaterial(node_bsdf, renderer);
		node_bsdf = node_bsdf->next_sibling("bsdf");
	}

	std::cout << "[info] load shapes..." << std::endl;
	auto node_shape = node_scene->first_node("shape");
	while (node_shape)
	{
		ParseShape(node_shape, renderer);
		node_shape = node_shape->next_sibling("shape");
	}

	auto envmap = ParseEnvmap(node_scene->first_node("emitter"));
	renderer->SetEnvmap(envmap);

	auto camera = ParseCamera(node_scene->first_node("sensor"));
	renderer->SetCamera(camera);

	auto integrator = ParseIntegrator(node_scene->first_node("integrator"));
	renderer->SetIntegrator(integrator);

	delete file_doc;
	delete xml_doc;

	return renderer;
}

Integrator *XmlParser::ParseIntegrator(rapidxml::xml_node<> *node_integrator)
{
	auto type = GetAttri(node_integrator, "type").value();
	auto max_depth = GetInt(node_integrator, "maxDepth", true).value_or(-1);
	auto rr_depth = GetInt(node_integrator, "rrDepth", true).value_or(5);
	switch (Hash(type.c_str()))
	{
	case "path"_hash:
		return new PathIntegrator(max_depth, rr_depth);
		break;
	case "bdpt"_hash:
		return new BdptIntegrator(max_depth, rr_depth);
		break;
	default:
		std::cerr << "[warning] " << GetTreeName(node_integrator) << std::endl
				  << "\tcannot handle integrator type" << type << "\"" << std::endl
				  << "use default \"path\" integrator" << std::endl;
		return new PathIntegrator(max_depth, rr_depth);
		break;
	}
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
	auto eye_pos = Vector3(0, 0, 0);
	auto look_at = Vector3(0, 0, 1);
	auto up = Vector3(0, 1, 0);
	if (auto node_lookat = GetChild(node_sensor, "toWorld")->first_node("lookat");
		node_lookat)
	{
		auto origin_str = GetAttri(node_lookat, "origin").value();
		sscanf(origin_str.c_str(), "%lf, %lf, %lf", &eye_pos[0], &eye_pos[1], &eye_pos[2]);

		auto target_str = GetAttri(node_lookat, "target").value();
		sscanf(target_str.c_str(), "%lf, %lf, %lf", &look_at[0], &look_at[1], &look_at[2]);

		auto up_str = GetAttri(node_lookat, "up").value();
		sscanf(up_str.c_str(), "%lf, %lf, %lf", &up[0], &up[1], &up[2]);
	}
	else if (auto to_world = GetToWorld(node_sensor);
			 to_world)
	{
		eye_pos = TransfromPt(*to_world, eye_pos);
		look_at = TransfromPt(*to_world, look_at);
		up = TransfromDir(*to_world, up);
	}

	auto film = ParseFilm(node_sensor);
	auto fov_height = fov_width * film.height / film.width;
	auto node_sampler = node_sensor->first_node("sampler");
	auto sample_count = GetInt(node_sampler, "sampleCount", false).value();
	return new Camera(film, eye_pos, look_at, up, fov_height, sample_count);
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

std::unique_ptr<Texture> XmlParser::ParseBumpMapping(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type)
{
	if (bsdf_type != "bumpmap")
		return nullptr;

	auto node_bump = node_bsdf->first_node("texture");
	auto bump_map = ParseTexture(node_bump);

	node_bsdf = node_bsdf->first_node("bsdf");
	if (!node_bsdf)
	{
		std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
				  << "\tnot enough bsdf information" << std::endl;
		exit(1);
	}
	bsdf_type = GetAttri(node_bsdf, "type").value();

	return bump_map;
}

std::unique_ptr<Texture> XmlParser::ParseOpacity(rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type)
{
	if (bsdf_type != "mask")
		return nullptr;

	auto node_opacity = GetChild(node_bsdf, "opacity", false);
	auto opacity_map = ParseTextureOrOther(node_opacity, "opacity");
	if (!opacity_map)
	{
		std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
				  << "\tnot enough opacity information" << std::endl;
		exit(1);
	}

	if (opacity_map->Constant() &&
		(!FloatEqual(opacity_map->Color(Vector2(0)).r, opacity_map->Color(Vector2(0)).g) ||
		 !FloatEqual(opacity_map->Color(Vector2(0)).r, opacity_map->Color(Vector2(0)).b) ||
		 !FloatEqual(opacity_map->Color(Vector2(0)).g, opacity_map->Color(Vector2(0)).b)))
	{
		std::cerr << "[error] " << GetTreeName(node_opacity) << std::endl
				  << "\tnot support different opacity for different color channel" << std::endl;
		exit(1);
	}

	node_bsdf = node_bsdf->first_node("bsdf");
	if (!node_bsdf)
	{
		std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
				  << "\tnot enough bsdf information" << std::endl;
		exit(1);
	}
	bsdf_type = GetAttri(node_bsdf, "type").value();

	return opacity_map;
}

bool XmlParser::ParseCoating(const std::string &id, rapidxml::xml_node<> *&node_bsdf, std::string &bsdf_type)
{
	if (bsdf_type != "coating" &&
		bsdf_type != "roughcoating")
		return false;
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
		if (!id_to_material_.count(ref_id))
		{
			std::cerr << "[error] " << GetTreeName(node_ref) << std::endl
					  << "\tcannot find existed material with id: " << ref_id << std::endl;
			exit(1);
		}
		else
		{
			id_to_material_[id] = id_to_material_[ref_id];
			return true;
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
	return false;
}

void XmlParser::ParseMaterial(rapidxml::xml_node<> *node_bsdf, Renderer *renderer, const std::string &id_default)
{
	auto bsdf_type = GetAttri(node_bsdf, "type").value();
	auto bump_map = ParseBumpMapping(node_bsdf, bsdf_type);
	auto id = id_default.empty() ? GetAttri(node_bsdf, "id").value() : id_default;
	auto opacity_map = ParseOpacity(node_bsdf, bsdf_type);
	auto twsided = false;
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
	if (ParseCoating(id, node_bsdf, bsdf_type))
		return;
	auto material = static_cast<Material *>(nullptr);
	switch (Hash(bsdf_type.c_str()))
	{
	case "diffuse"_hash:
		material = ParseDiffuse(node_bsdf);
		break;
	case "roughdiffuse"_hash:
		std::cout << "[warning] unsupported rough diffuse bsdf, treat as diffuse." << std::endl;
		material = ParseDiffuse(node_bsdf);
		break;
	case "dielectric"_hash:
		twsided = true;
		material = ParseDielectric(node_bsdf);
		break;
	case "roughdielectric"_hash:
		twsided = true;
		material = ParseRoughDielectric(node_bsdf);
		break;
	case "thindielectric"_hash:
		twsided = true;
		material = ParseDielectric(node_bsdf, true);
		break;
	case "conductor"_hash:
		material = ParseConductor(node_bsdf);
		break;
	case "roughconductor"_hash:
		material = ParseRoughConductor(node_bsdf);
		break;
	case "plastic"_hash:
		material = ParsePlastic(node_bsdf);
		break;
	case "roughplastic"_hash:
		material = ParseRoughPlastic(node_bsdf);
		break;
	default:
		std::cerr << "[error] " << GetTreeName(node_bsdf) << std::endl
				  << "\tcannot handle bsdf type " << bsdf_type << std::endl;
		exit(1);
	}

	material->SetTwosided(twsided);
	material->SetBumpMapping(std::move(bump_map));
	material->SetOpacity(std::move(opacity_map));
	id_to_material_[id] = material;
	renderer->AddMaterial(material);
}

Material *XmlParser::ParseDiffuse(rapidxml::xml_node<> *node_diffuse)
{
	auto reflectance = ParseTextureOrOther(node_diffuse, "reflectance");
	if (!reflectance)
		reflectance.reset(new ConstantTexture(Spectrum(0.5)));

	return new Diffuse(std::move(reflectance));
}

Material *XmlParser::ParseDielectric(rapidxml::xml_node<> *node_dielectric, bool thin)
{
	auto int_ior = GetIor(node_dielectric, "intIOR", "bk7");
	auto ext_ior = GetIor(node_dielectric, "extIOR", "air");
	auto specular_reflectance = ParseTextureOrOther(node_dielectric, "specularReflectance");
	auto specular_transmittance = ParseTextureOrOther(node_dielectric, "specularTransmittance");
	if (thin)
		return new ThinDielectric(int_ior,
								  ext_ior,
								  std::move(specular_reflectance),
								  std::move(specular_transmittance));
	else
		return new Dielectric(int_ior,
							  ext_ior,
							  std::move(specular_reflectance),
							  std::move(specular_transmittance));
}

Material *XmlParser::ParseRoughDielectric(rapidxml::xml_node<> *node_rough_dielectric)
{
	auto int_ior = GetIor(node_rough_dielectric, "intIOR", "bk7");
	auto ext_ior = GetIor(node_rough_dielectric, "extIOR", "air");

	auto specular_reflectance = ParseTextureOrOther(node_rough_dielectric, "specularReflectance");
	auto specular_transmittance = ParseTextureOrOther(node_rough_dielectric, "specularTransmittance");

	auto distri = GetString(node_rough_dielectric, "distribution").value_or("beckmann");

	auto alpha_u = ParseTextureOrOther(node_rough_dielectric, "alpha");
	if (!alpha_u)
		alpha_u = ParseTextureOrOther(node_rough_dielectric, "alphaU");
	if (!alpha_u)
		alpha_u.reset(new ConstantTexture(Spectrum(0.1)));
	auto alpha_v = ParseTextureOrOther(node_rough_dielectric, "alphaV");

	return new RoughDielectric(int_ior,
							   ext_ior,
							   std::move(specular_reflectance),
							   std::move(specular_transmittance),
							   GetDistrbType(distri),
							   std::move(alpha_u),
							   std::move(alpha_v));
}

Material *XmlParser::ParseConductor(rapidxml::xml_node<> *node_conductor)
{
	auto mirror = true;
	auto eta = Spectrum(0);
	auto k = Spectrum(1);
	auto ext_eta = GetIor(node_conductor, "extEta", "air");
	auto node_material = GetChild(node_conductor, "material");
	if (node_material)
	{
		auto material_name = GetAttri(node_material, "value").value();
		if (IOR_eta.count(material_name))
		{
			mirror = false;
			eta = IOR_eta.at(material_name),
			k = IOR_k.at(material_name);
		}
		else if (material_name != "none")
		{
			std::cerr << "[error] " << GetTreeName(node_material) << std::endl
					  << " unsupported material :" << material_name << ", "
					  << "use default Conductor material instead." << std::endl;
			exit(1);
		}
	}
	else if (node_conductor->first_node())
	{
		mirror = false;
		auto node_eta = GetChild(node_conductor, "eta", false);
		eta = GetSpectrum(node_eta);
		auto node_k = GetChild(node_conductor, "k", false);
		k = GetSpectrum(node_k);
	}

	auto specular_reflectance = ParseTextureOrOther(node_conductor, "specularReflectance");

	return new Conductor(mirror,
						 eta,
						 k,
						 ext_eta,
						 std::move(specular_reflectance));
}

Material *XmlParser::ParseRoughConductor(rapidxml::xml_node<> *node_rough_conductor)
{
	auto mirror = true;
	auto eta = Spectrum(0);
	auto k = Spectrum(1);
	auto ext_eta = GetIor(node_rough_conductor, "extEta", "air");
	auto node_material = GetChild(node_rough_conductor, "material");
	if (node_material)
	{
		auto material_name = GetAttri(node_material, "value").value();
		if (IOR_eta.count(material_name))
		{
			mirror = false;
			eta = IOR_eta.at(material_name),
			k = IOR_k.at(material_name);
		}
		else if (material_name != "none")
		{
			std::cerr << "[error] " << GetTreeName(node_material) << std::endl
					  << "unsupported material :" << material_name << ", "
					  << "use default Conductor material instead." << std::endl;
			exit(1);
		}
	}
	else if (node_rough_conductor->first_node())
	{
		mirror = false;
		auto node_eta = GetChild(node_rough_conductor, "eta", false);
		eta = GetSpectrum(node_eta);
		auto node_k = GetChild(node_rough_conductor, "k", false);
		k = GetSpectrum(node_k);
	}

	auto specular_reflectance = ParseTextureOrOther(node_rough_conductor, "specularReflectance");

	auto distri = GetString(node_rough_conductor, "distribution").value_or("beckmann");
	auto alpha_u = ParseTextureOrOther(node_rough_conductor, "alpha");
	if (!alpha_u)
		alpha_u = ParseTextureOrOther(node_rough_conductor, "alphaU");
	if (!alpha_u)
		alpha_u.reset(new ConstantTexture(Spectrum(0.1)));
	auto alpha_v = ParseTextureOrOther(node_rough_conductor, "alphaV");

	return new RoughConductor(mirror,
							  eta,
							  k,
							  ext_eta,
							  std::move(specular_reflectance),
							  GetDistrbType(distri),
							  std::move(alpha_u),
							  std::move(alpha_v));
}

Material *XmlParser::ParsePlastic(rapidxml::xml_node<> *node_plastic)
{
	auto int_ior = GetIor(node_plastic, "intIOR", "polypropylene");
	auto ext_ior = GetIor(node_plastic, "extIOR", "air");
	auto specular_reflectance = ParseTextureOrOther(node_plastic, "specularReflectance");
	auto diffuse_reflectance = ParseTextureOrOther(node_plastic, "diffuseReflectance");
	if (!diffuse_reflectance)
		diffuse_reflectance.reset(new ConstantTexture(Spectrum(0.5)));
	bool nonlinear = GetBoolean(node_plastic, "nonlinear").value_or(false);

	return new Plastic(int_ior,
					   ext_ior,
					   std::move(diffuse_reflectance),
					   std::move(specular_reflectance),
					   nonlinear);
}

Material *XmlParser::ParseRoughPlastic(rapidxml::xml_node<> *node_rough_plastic)
{
	auto int_ior = GetIor(node_rough_plastic, "intIOR", "polypropylene");
	auto ext_ior = GetIor(node_rough_plastic, "extIOR", "air");

	auto specular_reflectance = ParseTextureOrOther(node_rough_plastic, "specularReflectance");
	auto diffuse_reflectance = ParseTextureOrOther(node_rough_plastic, "diffuseReflectance");
	if (!diffuse_reflectance)
		diffuse_reflectance.reset(new ConstantTexture(0.5));

	auto distri = GetString(node_rough_plastic, "distribution").value_or("beckmann");
	auto alpha = ParseTextureOrOther(node_rough_plastic, "alpha");
	if (!alpha)
		alpha.reset(new ConstantTexture(Spectrum(0.1)));

	auto nonlinear = GetBoolean(node_rough_plastic, "nonlinear").value_or(false);

	return new RoughPlastic(int_ior,
							ext_ior,
							std::move(diffuse_reflectance),
							std::move(specular_reflectance),
							GetDistrbType(distri),
							std::move(alpha),
							nonlinear);
}

void XmlParser::ParseShape(rapidxml::xml_node<> *node_shape, Renderer *renderer)
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
		auto area_light = new AreaLight(radiance);
		renderer->AddMaterial(area_light);
		id_to_material_[ref] = area_light;
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
			ParseMaterial(node_bsdf, renderer, ref);
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

	if (!id_to_material_.count(ref))
	{
		std::cerr << "[error] " << GetTreeName(node_shape) << std::endl
				  << "\tcannot find material with name: \"" << ref << "\"" << std::endl;
		exit(1);
	}

	auto material = id_to_material_[ref];
	auto shape = static_cast<Shape *>(nullptr);
	switch (Hash(type.c_str()))
	{
	case "disk"_hash:
		shape = new Disk(material, std::move(to_world), flip_normals);
		break;
	case "sphere"_hash:
	{
		auto radius = GetFloat(node_shape, "radius").value_or(1);
		auto center = GetPoint(node_shape, "center").value_or(Vector3(0));
		shape = new Sphere(material, center, radius, std::move(to_world), flip_normals);
		break;
	}
	case "cube"_hash:
		shape = new Cube(material, std::move(to_world), flip_normals);
		break;
	case "rectangle"_hash:
		shape = new Rectangle(material, std::move(to_world), flip_normals);
		break;
	case "obj"_hash:
	{
		auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
		auto flip_tex_coords = GetBoolean(node_shape, "flipTexCoords").value_or(true);
		auto model_path = xml_directory_ + GetAttri(GetChild(node_shape, "filename", false), "value").value();
		shape = ModelParser::Parse(model_path, material, std::move(to_world), flip_normals, face_normals, flip_tex_coords);
		break;
	}
	case "ply"_hash:
	{
		auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
		auto model_path = xml_directory_ + GetAttri(GetChild(node_shape, "filename", false), "value").value();
		shape = ModelParser::Parse(model_path, material, std::move(to_world), flip_normals, face_normals);
		break;
	}
	default:
		std::cerr << "[error] " << GetTreeName(node_shape) << std::endl
				  << "\tcannot handle shape type " << std::endl;
		exit(1);
	}
	renderer->AddShape(shape);
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

std::unique_ptr<Texture> XmlParser::ParseTextureOrOther(rapidxml::xml_node<> *node_parent, std::string name)
{
	auto node = GetChild(node_parent, name, true);
	if (!node)
		return nullptr;

	auto node_type = node->name();
	switch (Hash(node_type))
	{
	case "float"_hash:
	{
		auto value = std::stof(GetAttri(node, "value").value());
		return std::make_unique<ConstantTexture>(Spectrum(value));
		break;
	}
	case "rgb"_hash:
	case "spectrum"_hash:
	{
		auto value = GetSpectrum(node);
		return std::make_unique<ConstantTexture>(value);
		break;
	}
	case "texture"_hash:
	{
		return ParseTexture(node);
		break;
	}
	default:
		std::cerr << "[error] " << GetTreeName(node) << std::endl
				  << "\tcannot handle texture" << std::endl;
		exit(1);
	}
	return nullptr;
}

std::unique_ptr<Texture> XmlParser::ParseTexture(rapidxml::xml_node<> *node_texture)
{
	auto texture_type = GetAttri(node_texture, "type").value();
	switch (Hash(texture_type.c_str()))
	{
	case "bitmap"_hash:
	{
		auto node_filename = GetChild(node_texture, "filename", false);
		auto img_path = xml_directory_ + ConvertBackSlash(GetAttri(node_filename, "value").value());
		auto gamma = GetFloat(node_texture, "gamma").value_or(gamma_);
		return std::make_unique<Bitmap>(img_path, gamma);
		break;
	}
	case "checkerboard"_hash:
	{
		auto node_color0 = GetChild(node_texture, "color0");
		auto color0 = node_color0 ? GetSpectrum(node_color0) : Spectrum(0.4);

		auto node_color1 = GetChild(node_texture, "color1");
		auto color1 = node_color1 ? GetSpectrum(node_color1) : Spectrum(0.2);

		auto u_offset = GetFloat(node_texture, "uoffset", false).value(),
			 v_offset = GetFloat(node_texture, "voffset", false).value(),
			 u_scale = GetFloat(node_texture, "uscale", false).value(),
			 v_scale = GetFloat(node_texture, "vscale", false).value();

		return std::make_unique<Checkerboard>(color0, color1, Vector2(u_offset, v_offset), Vector2(u_scale, v_scale));
		break;
	}
	case "gridtexture"_hash:
	{
		auto node_color0 = GetChild(node_texture, "color0");
		auto color0 = node_color0 ? GetSpectrum(node_color0) : Spectrum(0.4);

		auto node_color1 = GetChild(node_texture, "color1");
		auto color1 = node_color1 ? GetSpectrum(node_color1) : Spectrum(0.2);

		auto line_width = GetFloat(node_texture, "lineWidth").value_or(0.01),
			 u_offset = GetFloat(node_texture, "uoffset", false).value(),
			 v_offset = GetFloat(node_texture, "voffset", false).value(),
			 u_scale = GetFloat(node_texture, "uscale", false).value(),
			 v_scale = GetFloat(node_texture, "vscale", false).value();

		return std::make_unique<GridTexture>(color0, color1, line_width, Vector2(u_offset, v_offset), Vector2(u_scale, v_scale));
		break;
	}

	default:
		std::cerr << "[error] " << GetTreeName(node_texture) << std::endl
				  << "\tcannot handle texture type except from bitmap" << std::endl;
		exit(1);
		break;
	}
	return nullptr;
}

std::optional<std::string> XmlParser::GetAttri(rapidxml::xml_node<> *node, std::string key, bool not_exist_ok)
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

rapidxml::xml_node<> *XmlParser::GetChild(rapidxml::xml_node<> *node, std::string name, bool not_exist_ok)
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

std::unique_ptr<Mat4> XmlParser::GetToWorld(rapidxml::xml_node<> *node_parent)
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

	auto result = Mat4(1);
	sscanf(matrix_str.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
		   &result[0][0], &result[1][0], &result[2][0], &result[3][0],
		   &result[0][1], &result[1][1], &result[2][1], &result[3][1],
		   &result[0][2], &result[1][2], &result[2][2], &result[3][2],
		   &result[0][3], &result[1][3], &result[2][3], &result[3][3]);

	if (glm::isIdentity(result, kEpsilon))
		return nullptr;
	else
		return std::make_unique<Mat4>(result);
}

Spectrum XmlParser::GetSpectrum(rapidxml::xml_node<> *node_spectrum)
{
	if (strcmp(node_spectrum->name(), "rgb") != 0)
	{
		std::cerr << "[error] " << GetTreeName(node_spectrum) << std::endl
				  << "\tcannot hanle  spectrum except from rgb" << std::endl;
		exit(1);
	}
	auto value_str = GetAttri(node_spectrum, "value").value();
	Spectrum result;

	sscanf(value_str.c_str(), "%lf, %lf, %lf", &result[0], &result[1], &result[2]);

	return result;
}

std::optional<bool> XmlParser::GetBoolean(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
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

std::optional<std::string> XmlParser::GetString(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
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

std::optional<int> XmlParser::GetInt(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
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

std::optional<Float> XmlParser::GetFloat(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
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

Float XmlParser::GetIor(rapidxml::xml_node<> *node_parent, std::string ior_type, std::string default_material_name)
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

std::string XmlParser::GetTreeName(rapidxml::xml_node<> *node)
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

std::optional<Vector3> XmlParser::GetPoint(rapidxml::xml_node<> *node_parent, std::string name, bool not_exist_ok)
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

MicrofacetDistribType XmlParser::GetDistrbType(const std::string &name)
{
	switch (Hash(name.c_str()))
	{
	case "beckmann"_hash:
		return MicrofacetDistribType::kBeckmann;
		break;
	case "ggx"_hash:
		return MicrofacetDistribType::kGgx;
		break;
	default:
		std::cout << "[warning] unkown microfacet distribution: " << name << ", use Beckmann instead.";
		return MicrofacetDistribType::kBeckmann;
		break;
	}
}

NAMESPACE_END(simple_renderer)