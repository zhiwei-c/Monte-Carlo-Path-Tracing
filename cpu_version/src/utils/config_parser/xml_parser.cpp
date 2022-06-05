#include "xml_parser.h"

#include <map>
#include <string>
#include <cstring>
#include <iostream>
#include <set>

#include "rapidxml/rapidxml_utils.hpp"
#include "glm/gtx/matrix_query.hpp"

NAMESPACE_BEGIN(raytracer)

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
		ParseBsdf(node_bsdf, renderer);
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
		renderer->AddBsdf(area_light);
		id_to_bsdf_[ref] = area_light;
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
			ParseBsdf(node_bsdf, renderer, ref);
	}
	auto to_world = GetToWorld(node_shape);
	auto type = GetAttri(node_shape, "type").value();
	auto flip_normals = GetBoolean(node_shape, "flipNormals").value_or(false);

	Bsdf *bsdf = nullptr;
	if (id_to_bsdf_.count(ref))
	{
		bsdf = id_to_bsdf_[ref];
	}

	Medium *int_medium = nullptr, *ext_medium = nullptr;
	auto node_medium = node_shape->first_node("medium");
	while (node_medium)
	{
		std::string medium_id = "unnamed_medium_" + std::to_string(media_cnt_++);

		bool interior = ParseMedium(node_medium, renderer, medium_id);
		if (interior)
			int_medium = id_to_medium_[medium_id];
		else
			ext_medium = id_to_medium_[medium_id];
		node_medium = node_medium->next_sibling("medium");
	}

	if (ref.empty() && !int_medium && !ext_medium)
	{
		std::cerr << "[error] " << GetTreeName(node_shape) << std::endl
				  << "\tcannot find supported bsdf or medium info" << std::endl;
	}

	Shape *shape = nullptr;
	switch (Hash(type.c_str()))
	{
	case "disk"_hash:
		shape = new Disk(bsdf, int_medium, ext_medium, std::move(to_world), flip_normals);
		break;
	case "sphere"_hash:
	{
		auto radius = GetFloat(node_shape, "radius").value_or(1);
		auto center = GetPoint(node_shape, "center").value_or(Vector3(0));
		shape = new Sphere(bsdf, int_medium, ext_medium, center, radius, std::move(to_world), flip_normals);
		break;
	}
	case "cube"_hash:
		shape = new Cube(bsdf, int_medium, ext_medium, std::move(to_world), flip_normals);
		break;
	case "rectangle"_hash:
		shape = new Rectangle(bsdf, int_medium, ext_medium, std::move(to_world), flip_normals);
		break;
	case "obj"_hash:
	{
		auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
		auto flip_tex_coords = GetBoolean(node_shape, "flipTexCoords").value_or(true);
		auto model_path = xml_directory_ + GetAttri(GetChild(node_shape, "filename", false), "value").value();
		shape = ModelParser::Parse(model_path, bsdf, int_medium, ext_medium, std::move(to_world), flip_normals, face_normals, flip_tex_coords);
		break;
	}
	case "ply"_hash:
	{
		auto face_normals = GetBoolean(node_shape, "faceNormals").value_or(false);
		auto model_path = xml_directory_ + GetAttri(GetChild(node_shape, "filename", false), "value").value();
		shape = ModelParser::Parse(model_path, bsdf, int_medium, ext_medium, std::move(to_world), flip_normals, face_normals);
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

NAMESPACE_END(raytracer)