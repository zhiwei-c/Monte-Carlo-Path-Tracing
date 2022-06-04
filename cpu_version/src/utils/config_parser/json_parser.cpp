#include "json_parser.h"

#include <fstream>
#include <iostream>

NAMESPACE_BEGIN(raytracer)

Renderer *ParseJsonCfg(const std::string &file_path)
{
	std::ifstream in(file_path);
	std::string dir_path = GetDirectory(file_path);
	if (!in)
	{
		std::cerr << "[error] "
				  << "open " << file_path << " failed" << std::endl;
		exit(1);
	}
	std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	in.close();

	auto json_data = nlohmann::json::parse(contents.c_str());

	auto renderer = new Renderer();

	auto model_name = GetString(json_data, "scene", false).value();
	renderer->AddShapesBsdfs(dir_path + model_name);

	auto envmap = InitEnvmap(dir_path, json_data);
	renderer->SetEnvmap(envmap);

	auto integrator = InitIntegrator(json_data);
	renderer->SetIntegrator(integrator);

	auto camera = InitCamera(json_data);
	renderer->SetCamera(camera);

	return renderer;
}

Integrator *InitIntegrator(nlohmann::json &data)
{
	std::string type = "path";
	int max_depth = -1;
	int rr_depth = 5;
	if (!data.contains("integrator"))
	{
		std::cout << "[warning] "
				  << "missing \"integrator\", use default integrator" << std::endl;
	}
	else if (!data["integrator"].is_object())
	{
		std::cerr << "[error] "
				  << "error format for \"integrator\"" << std::endl;
		exit(1);
	}
	else
	{
		type = GetString(data["integrator"], "type", true).value_or("path");
		max_depth = GetInt(data["integrator"], "max_depth", true).value_or(-1);
		rr_depth = GetInt(data["integrator"], "rr_depth", true).value_or(5);
	}

	Integrator *integrator = nullptr;
	switch (Hash(type.c_str()))
	{
	case "path"_hash:
		integrator = new PathIntegrator(max_depth, rr_depth);
		break;
	case "bdpt"_hash:
		integrator = new BdptIntegrator(max_depth, rr_depth);
		break;
	default:
		std::cerr << "[error] cannot handle integrator type" << type << "\"" << std::endl;
		exit(1);
		break;
	}

	return integrator;
}

Camera *InitCamera(nlohmann::json &data)
{
	if (!data.contains("camera"))
	{
		std::cerr << "[error] "
				  << "missing \"camera\"" << std::endl;
		exit(1);
	}
	if (!data["camera"].is_object())
	{
		std::cerr << "[error] "
				  << "error format for \"camera\"" << std::endl;
		exit(1);
	}
	auto eye_pos = GetVec3(data["camera"], "eye_pos", false).value();
	auto look_at = GetVec3(data["camera"], "look_at", false).value();
	auto up = GetVec3(data["camera"], "up", false).value();
	auto fov_height = GetFloat(data["camera"], "fov_height", false).value();
	auto spp = GetInt(data["camera"], "spp", false).value();
	auto film = InitFilm(data);
	return new Camera(film, eye_pos, look_at, up, fov_height, spp);
}

Film InitFilm(nlohmann::json &data)
{
	auto film = Film();

	film.width = GetInt(data["camera"], "width", false).value();
	film.height = GetInt(data["camera"], "height", false).value();
	film.gamma = GetFloat(data["camera"], "gamma").value_or(2.2);
	film.format = GetString(data["camera"], "format").value_or("png");
	return film;
}

Envmap *InitEnvmap(const std::string &dir_path, nlohmann::json &data)
{
	if (!data.contains("envmap"))
	{
		return nullptr;
	}
	if (!data["envmap"].is_object())
	{
		std::cerr << "[error] "
				  << "error format for \"sky_box\"" << std::endl;
		exit(1);
	}

	if (data["envmap"].empty())
		return new Envmap();

	auto path = dir_path + GetString(data["envmap"], "path").value_or("");
	return new Envmap(path, 1);
}

std::optional<Vector3> GetVec3(const nlohmann::json &data, const std::string &name, bool not_exist_ok)
{
	if (!data.contains(name))
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] "
					  << "missing \"" + name + "\"" << std::endl;
			exit(1);
		}
	}
	if (!data[name].is_array() || data[name].size() != 3)
	{
		std::cerr << "[error] "
				  << "error format for \"" + name + "\"" << std::endl;
		exit(1);
	}

	auto res = Vector3(0);
	for (int i = 0; i < 3; i++)
	{
		if (!data[name][i].is_number())
		{
			std::cerr << "[error] "
					  << "error format for \"" + name + "\"" << std::endl;
			exit(1);
		}
		else
		{
			res[i] = static_cast<Float>(data[name][i]);
		}
	}
	return res;
}

std::optional<std::string> GetString(const nlohmann::json &data, const std::string &name, bool not_exist_ok)
{
	if (!data.contains(name))
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] "
					  << "missing \"" + name + "\"" << std::endl;
			exit(1);
		}
	}
	if (!data[name].is_string())
	{
		std::cerr << "[error] "
				  << "error format for \"" + name + "\"" << std::endl;
		exit(1);
	}
	return static_cast<std::string>(data[name]);
}

std::optional<int> GetInt(const nlohmann::json &data, const std::string &name, bool not_exist_ok)
{
	if (!data.contains(name))
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] "
					  << "missing \"" + name + "\"" << std::endl;
			exit(1);
		}
	}
	if (!data[name].is_number_integer())
	{
		std::cerr << "[error] "
				  << "error format for \"" + name + "\"" << std::endl;
		exit(1);
	}
	return static_cast<int>(data[name]);
}

std::optional<Float> GetFloat(const nlohmann::json &data, const std::string &name, bool not_exist_ok)
{
	if (!data.contains(name))
	{
		if (not_exist_ok)
			return std::nullopt;
		else
		{
			std::cerr << "[error] "
					  << "missing \"" + name + "\"" << std::endl;
			exit(1);
		}
	}
	if (!data[name].is_number())
	{
		std::cerr << "[error] "
				  << "error format for \"" + name + "\"" << std::endl;
		exit(1);
	}
	return static_cast<Float>(data[name]);
}

NAMESPACE_END(raytracer)