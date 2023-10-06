#include <filesystem>
#include <iostream>

#include "utils/file_path.hpp"
#include "utils/config_parser.hpp"

int main(int argc, char *argv[])
{
    // usage example: .\SimpleRenderer.exe  render_config.xml [output_path.png]
    if (argc > 3 || argc < 2)
    {
        std::cerr << "[error] incorrect argument num" << std::endl;
        exit(1);
    }

    //绘制图像保存路径
    std::string output_filename = "result.png";
    if (argc == 3)
    {
        output_filename = argv[2];
        auto out_directory = raytracer::GetDirectory(argv[2]);
        if (!out_directory.empty() && !std::filesystem::exists(std::filesystem::path(out_directory)))
        {
            std::cerr << "[error] invalid output directory :" << out_directory << std::endl;
            exit(1);
        }
    }

    //绘制图像配置文件路径
    std::filesystem::path file_path = argv[1];
    if (raytracer::GetSuffix(argv[1]) != "xml")
    {
        std::cerr << "[error] only support mitsuba format config file" << std::endl;
        exit(1);
    }
    else if (!std::filesystem::exists(file_path))
    {
        std::cerr << "[error] cannot find config file: \"" << argv[1] << "\"" << std::endl;
        exit(1);
    }
    std::cout << "[info] read config file: \"" << argv[1] << "\"" << std::endl;

    raytracer::Renderer renderer;
    raytracer::LoadMitsubaConfig(argv[1], renderer);
    renderer.Render(output_filename);

    return 0;
}