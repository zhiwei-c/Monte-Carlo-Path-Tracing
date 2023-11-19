#include <cstdio>
#include <iostream>

#include "ray_tracer.cuh"

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
    csrt::Config confg;
    try
    {
        confg = csrt::LoadConfig(param.input);
    }
    catch (const std::exception &e)
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

    RayTracer *ray_tracer = nullptr;
    try
    {
        ray_tracer = new RayTracer(confg);
    }
    catch (const std::exception &e)
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
    catch (const std::exception &e)
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
                 argv[i] == std::string("--g"))
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
