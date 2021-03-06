#include <filesystem>

#include "utils/config_parser/xml_parser.h"
#include "utils/config_parser/json_parser.h"

int main(int argc, char *argv[])
{
	// usage example: .\SimpleRenderer.exe  render_config.xml [output_path.png]
	if (argc > 3 || argc < 2)
	{
		std::cerr << "[error] incorrect argument num" << std::endl;
		exit(1);
	}

	//绘制图像保存路径
	std::string output_name = "";
	if (argc == 3)
	{
		output_name = raytracer::ConvertBackSlash(argv[2]);
		auto out_directory = raytracer::GetDirectory(argv[2]);
		if (!out_directory.empty() &&
			!std::filesystem::exists(std::filesystem::path(out_directory)))
		{
			std::cerr << "[error] invalid output directory :" << out_directory << std::endl;
			exit(1);
		}
	}

	//绘制图像配置文件路径
	std::string file_path = raytracer::ConvertBackSlash(argv[1]);
	std::filesystem::path file_path_now = file_path;
	if (!std::filesystem::exists(file_path_now))
	{
		std::cerr << "[error] " << file_path << std::endl
				  << "\tcannot find file." << std::endl;
		exit(1);
	}
	std::cout << "[info] read config file: \"" << file_path.c_str() << "\"" << std::endl;

	//解析绘制图像配置文件
	raytracer::Renderer * renderer = nullptr;
	std::string suffix = raytracer::GetSuffix(file_path);
	if (suffix == "json")
		renderer = raytracer::ParseJsonCfg(file_path);
	else if (suffix == "xml")
	{
		auto parser = raytracer::XmlParser();
		renderer = parser.Parse(file_path);
	}
	else
	{
		std::cerr << "[error] " << file_path << std::endl
				  << "\tunsupported config format." << std::endl;
		exit(1);
	}

	//生成图像
	renderer->Render(output_name);

	delete renderer;
	renderer = nullptr;
	return 0;
}
