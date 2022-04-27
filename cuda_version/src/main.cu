#include <iostream>
#include <string>
#include <filesystem>

#include "utils/config_parser.h"

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
		output_name = ConvertBackSlash(argv[2]);
		auto out_directory = GetDirectory(argv[2]);
		if (!out_directory.empty() &&
			!std::filesystem::exists(std::filesystem::path(out_directory)))
		{
			std::cerr << "[error] invalid output directory :" << out_directory << std::endl;
			exit(1);
		}
	}
	if (output_name.empty())
	{
		std::cerr << "[warning] empty output filename, use default \"result.png\"" << std::endl;
		output_name = "result.png";
	}

	//绘制图像配置文件路径
	auto file_path = ConvertBackSlash(argv[1]);
	std::filesystem::path file_path_now = file_path;
	if (!std::filesystem::exists(file_path_now))
	{
		std::cerr << "[error] " << file_path << std::endl
				  << "\tcannot find file." << std::endl;
		exit(1);
	}
	std::cout << "[info] read config file: \"" << file_path.c_str() << "\"" << std::endl;

	auto suffix = GetSuffix(file_path);
	if (suffix != "xml")
	{
		std::cerr << "[error] " << file_path << std::endl
				  << "\tunsupported config format." << std::endl
				  << "\tonly support mitsuba1 xml file" << std::endl;
		exit(1);
	}

	auto renderer = ParseRenderConfig(file_path);
	renderer->Render(output_name);
	delete renderer;
	renderer = nullptr;
	return 0;
}
