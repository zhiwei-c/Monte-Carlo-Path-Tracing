#include <filesystem>

#include "utils/config_parser/json_parser.h"
#include "utils/config_parser/xml_parser.h"
#include "utils/timer.h"
#include "utils/file_path.h"

int main(int argc, char *argv[])
{

	if (argc > 3 || argc < 2)
	{
		std::cerr << "[error] incorrect argument num" << std::endl;
		exit(1);
	}

	std::string output_name = "";
	if (argc == 3){
		output_name = simple_renderer::ConvertBackSlash(argv[2]);
		auto out_directory = std::filesystem::path(simple_renderer::GetDirectory(argv[2]));
		if (!std::filesystem::exists(out_directory))
		{
			std::cerr << "[error] invalid output directory :" << out_directory << std::endl;
			exit(1);
		}
	}

	auto file_path = simple_renderer::ConvertBackSlash(argv[1]);
	std::filesystem::path file_path_now = file_path;
	if (!std::filesystem::exists(file_path_now))
	{
		std::cerr << "[error] " << file_path << std::endl
				  << "\tcannot find file." << std::endl;
		exit(1);
	}
	std::cout << "[info] read config file: \"" << file_path.c_str() << "\"" << std::endl;

	simple_renderer::Scene *scene = nullptr;
	simple_renderer::Camera *camera = nullptr;
	auto suffix = simple_renderer::GetSuffix(file_path);
	if (suffix == "json")
		std::tie(scene, camera) = simple_renderer::ParseJsonCfg(file_path);
	else if (suffix == "xml")
	{
		auto parser = simple_renderer::XmlParser();
		std::tie(scene, camera) = parser.Parse(file_path);
	}
	else
	{
		std::cerr << "[error] " << file_path << std::endl
				  << "\tunsupported config format." << std::endl;
		exit(1);
	}

	camera->Shoot(scene, output_name);

	delete scene;
	delete camera;

	scene = nullptr;
	camera = nullptr;

	return 0;
}
