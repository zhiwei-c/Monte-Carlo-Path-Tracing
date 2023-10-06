#include <string>
#include <filesystem>
#include <cstdio>

#include "painters/painters.cuh"
#include "utils/misc.cuh"

struct Param
{
    bool is_cpu;
    std::string input;
    std::string output;
    BvhBuilder::Type bvh_type;

    Param() : is_cpu(true), input(""), output("result.png"), bvh_type(BvhBuilder::Type::kLinear) {}
};

Param ParseParam(int argc, char **argv);
void PrintHelp();

int main(int argc, char **argv)
{
    Param param = ParseParam(argc, argv);

    ConfigParser config_parser;
    SceneInfo scene_info = config_parser.LoadConfig(param.input);

#ifdef ENABLE_CUDA
    Painter *painter = nullptr;
    if (param.is_cpu)
        painter = new CpuPainter(param.bvh_type, scene_info);
    else
        painter = new CudaPainter(param.bvh_type, scene_info);
#else
    Painter *painter = new CpuPainter(param.bvh_type, scene_info);
#endif

    painter->Draw(param.output);
    SAFE_DELETE_ELEMENT(painter);
    SAFE_DELETE_ELEMENT(scene_info.env_map);

    return 0;
}

Param ParseParam(int argc, char **argv)
{
    PrintHelp();

    Param param;
    for (int i = 0; i < argc; ++i)
    {
        if (argv[i] == std::string("--cpu"))
        {
            param.is_cpu = true;
        }
        else if (argv[i] == std::string("--cuda"))
        {
            param.is_cpu = false;
        }
        else if (argv[i] == std::string("--bvh") && i + 1 < argc)
        {
            if (argv[i + 1] == std::string("normal"))
            {
                param.bvh_type = BvhBuilder::Type::kNormal;
            }
            else if (argv[i + 1] == std::string("linear"))
            {
                param.bvh_type = BvhBuilder::Type::kLinear;
            }
        }
        else if (argv[i] == std::string("--input") && i + 1 < argc)
        {
            param.input = argv[i + 1];
        }
        else if (argv[i] == std::string("--output") && i + 1 < argc)
        {
            param.output = argv[i + 1];
        }
        else if(argv[i] == std::string("-h"  )||argv[i] == std::string("--help"))
        {
            exit(0);
        }
    }

    if (GetSuffix(param.output) != "png")
    {
        fprintf(stderr, "[warning] only support png output, ignore output format \"%s\".", GetSuffix(param.output).c_str());
        param.output = param.output.substr(0, param.output.find_last_of(".")) + ".png";
    }

    return param;
}

void PrintHelp()
{
    fprintf(stderr, "\nSimple Ray Tracer.\n");
    fprintf(stderr, "Command Format: [--cpu/--cuda] [--bvh 'bvh type'] [--input 'config path'] [--output 'file path]\n");
    fprintf(stderr, "--cpu: use CPU for rendering. if not specify CPU or CUDA, use CPU.\n");
    fprintf(stderr, "--cuda: use CUDA for rendering, no effect if disbale CUDA when compiling. if not specify CPU or CUDA, use CPU.\n");
    fprintf(stderr, "--bvh: bvh type for ray tracing, available: [linear, normal], default: linear\n");
    fprintf(stderr, "--input: read config from mitsuba format xml file, load default config if empty, default: empty\n");
    fprintf(stderr, "--output: output path for rendering result, only PNG format, default: 'result.png'\n");
    fprintf(stderr, "\n");
}