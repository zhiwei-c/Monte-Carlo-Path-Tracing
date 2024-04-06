#include <cstdio>
#include <iostream>

#include "csrt/ray_tracer.hpp"

struct Param
{
    csrt::BackendType type;
    bool preview;
    int width;
    int height;
    int sample_count;
    std::string input;
    std::string output;

    Param()
        : type(csrt::BackendType::kCpu), preview(false), width(0), height(0),
          sample_count(0), input(""), output("result.png")
    {
    }
};

Param ParseParam(int argc, char **argv);

int main(int argc, char **argv)
{
#ifdef ENABLE_CUDA
    cudaDeviceReset();
#endif

    Param param = ParseParam(argc, argv);
    csrt::RendererConfig confg;
    try
    {
        confg = csrt::LoadConfig(param.input);
    }
    catch (const csrt::MyException &e)
    {
        std::cerr << e.what() << '\n';
#ifdef ENABLE_CUDA
        cudaDeviceReset();
#endif
        return 0;
    }

    confg.backend_type = param.type;
    if (param.width > 0)
        confg.camera.width = param.width;
    if (param.height > 0)
        confg.camera.height = param.height;
    if (param.sample_count > 0)
        confg.camera.spp = param.sample_count;
    if (param.preview)
    {
        fprintf(
            stderr,
            "[info] spp is forced to be set to 1 when real-time previewing.\n");
        confg.camera.spp = 1;
    }

    csrt::RayTracer *ray_tracer = nullptr;
    try
    {
        ray_tracer = new csrt::RayTracer(confg);
    }
    catch (const csrt::MyException &e)
    {
        std::cerr << e.what() << '\n';
#ifdef ENABLE_CUDA
        cudaDeviceReset();
#endif
        return 0;
    }

    try
    {
#ifdef ENABLE_VIEWER
        if (param.preview)
            ray_tracer->Preview(argc, argv, param.output);
        else
#endif
            ray_tracer->Draw(param.output);
    }
    catch (const csrt::MyException &e)
    {
        std::cerr << e.what() << '\n';
        delete ray_tracer;
#ifdef ENABLE_CUDA
        cudaDeviceReset();
#endif
        return 0;
    }

    delete ray_tracer;
    return 0;
}

Param ParseParam(int argc, char **argv)
{
    std::cerr << "A Simple Ray Tracer.\n\n";
    std::cerr << "Command Format:\n";
    std::cerr << "  '[-c/--cpu/-g/--gpu/-p/--preview] "
                 "--input/-i 'config path' "
                 "[--output/-o 'file path] "
                 "[--width/-w 'value'] "
                 "[--height/-h 'value'] "
                 "[--spp/-s 'value']'.\n\n";
    std::cerr << "Option:\n";
    std::cerr << "  --'cpu' or '-c': use CPU for offline rendering.\n"
                 "      if not specify specify CPU/CUDA/preview, use CPU.\n";
    std::cerr << "  --'gpu' or '-g': use CUDA for offline rendering,\n"
                 "      no effect if disbale CUDA when compiling.\n"
                 "      if not specify specify CPU/CUDA/preview, use CPU.\n";
    std::cerr << "  --'preview' or '-p': use CUDA for real-time rendering,\n";
    std::cerr << "      no effect if disbale CUDA when compiling.\n";
    std::cerr << "      if not specify specify CPU/CUDA/preview, use CPU.\n";
    std::cerr
        << "  '--input' or '-i': read config from mitsuba format xml file.\n";
    std::cerr << "  '--output' or '-o': output path for rendering result\n"
                 "      only PNG format, default: 'result.png'.\n";
    std::cerr << "      press 's' key to save when real-time previewing.\n";
    std::cerr
        << "  '--width' or '-w': specify the width of rendering picture.\n";
    std::cerr
        << "  '--height' or '-h': specify the height of rendering picture.\n";
    std::cerr
        << "  '--spp' or '-s': specify the number of samples per pixel.\n\n";

    Param param;
    for (int i = 0; i < argc; ++i)
    {
        if (argv[i] == std::string("--cpu") || argv[i] == std::string("-c"))
        {
            param.type = csrt::BackendType::kCpu;
            param.preview = false;
        }
#ifdef ENABLE_CUDA
        else if (argv[i] == std::string("--gpu") ||
                 argv[i] == std::string("-g"))
        {
            param.type = csrt::BackendType::kCuda;
            param.preview = false;
        }
        else if ((argv[i] == std::string("--preview") ||
                  argv[i] == std::string("-p")))
        {
            param.type = csrt::BackendType::kCuda;
            param.preview = true;
        }
#endif
        else if ((argv[i] == std::string("--width") ||
                  argv[i] == std::string("-w")) &&
                 i + 1 < argc)
        {
            param.width = std::atoi(argv[i + 1]);
        }
        else if ((argv[i] == std::string("--height") ||
                  argv[i] == std::string("-h")) &&
                 i + 1 < argc)
        {
            param.height = std::atoi(argv[i + 1]);
        }
        else if ((argv[i] == std::string("--spp") ||
                  argv[i] == std::string("-s")) &&
                 i + 1 < argc)
        {
            param.sample_count = std::atoi(argv[i + 1]);
        }
        else if ((argv[i] == std::string("--input") ||
                  argv[i] == std::string("-i")) &&
                 i + 1 < argc)
        {
            param.input = argv[i + 1];
        }
        else if ((argv[i] == std::string("--output") ||
                  argv[i] == std::string("-o")) &&
                 i + 1 < argc)
        {
            param.output = argv[i + 1];
        }
        else if (argv[i] == std::string("--help"))
        {
            exit(0);
        }
    }

    std::string suffix = csrt::GetSuffix(param.output);
    if (suffix != "png")
    {
        fprintf(
            stderr,
            "[warning] only support png output, ignore output format \"%s\".",
            suffix.c_str());
        param.output =
            param.output.substr(0, param.output.find_last_of(".")) + ".png";
    }

    return param;
}
