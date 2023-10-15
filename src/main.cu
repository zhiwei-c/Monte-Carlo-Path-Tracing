#include <string>
#include <filesystem>
#include <cstdio>

#include "painters/painters.cuh"
#include "utils/misc.cuh"

enum class UsageType
{
    kCpuOffline,
    kCudaOffline,
    kCudaRealtime,
};

struct Param
{
    UsageType type;
    std::string input;
    std::string output;
    BvhBuilder::Type bvh_type;

    Param() : type(UsageType::kCpuOffline), input(""), output("result.png"),
              bvh_type(BvhBuilder::Type::kLinear) {}
};

Param ParseParam(int argc, char **argv);
void PrintHelp();

int main(int argc, char **argv)
{
    Param param = ParseParam(argc, argv);

    ConfigParser config_parser;
    SceneInfo scene_info = config_parser.LoadConfig(param.type == UsageType::kCudaRealtime, param.input);

    Painter *painter = nullptr;
    switch (param.type)
    {
#ifdef ENABLE_CUDA
    case UsageType::kCudaOffline:
        painter = new CudaPainter(param.bvh_type, scene_info);
        break;
#ifdef ENABLE_VIEWER
    case UsageType::kCudaRealtime:
        painter = new CudaViewer(argc, argv, param.bvh_type, scene_info);
        break;
#endif
#endif
    default:
        painter = new CpuPainter(param.bvh_type, scene_info);
        break;
    }

    painter->Draw(param.output);

    SAFE_DELETE_ELEMENT(painter);
    SAFE_DELETE_ELEMENT(scene_info.env_map);
    SAFE_DELETE_ELEMENT(scene_info.sun);

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
            param.type = UsageType::kCpuOffline;
        }
        else if (argv[i] == std::string("--gpu"))
        {
            param.type = UsageType::kCudaOffline;
        }
        else if (argv[i] == std::string("--preview"))
        {
            param.type = UsageType::kCudaRealtime;
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
        else if (argv[i] == std::string("-h") || argv[i] == std::string("--help"))
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
    fprintf(stderr, "Command Format: [--cpu/--gpu/--preview] [--bvh 'bvh type'] [--input 'config path'] [--output 'file path]\n");
    fprintf(stderr, "--cpu: use CPU for offline rendering. if not specify CPU/CUDA/preview, use CPU.\n");
    fprintf(stderr, "--gpu: use CUDA for offline rendering, no effect if disbale CUDA when compiling. if not specify CPU/CUDA/preview, use CPU.\n");
    fprintf(stderr, "--preview: use CUDA for real-time rendering, no effect if disbale CUDA when compiling. if not specify CPU/CUDA/preview, use CPU.\n");
    fprintf(stderr, "--bvh: bvh type for ray tracing, available: [linear, normal], default: linear\n");
    fprintf(stderr, "--input: read config from mitsuba format xml file, load default config if empty, default: empty\n");
    fprintf(stderr, "--output: output path for rendering result, only PNG format, press 's' key to save when real-time previewing, default: 'result.png'\n");
    fprintf(stderr, "\n");
}