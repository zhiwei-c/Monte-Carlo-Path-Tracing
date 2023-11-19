#include <cstdio>
#include <iostream>

#include "ray_tracer.cuh"

struct Param
{
    csrt::BackendType type;
    std::string input;
    std::string output;

    Param() : type(csrt::BackendType::kCpu), input(""), output("result.png") {}
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
        }
        else if (argv[i] == std::string("--gpu") ||
                 argv[i] == std::string("--g"))
        {
            param.type = csrt::BackendType::kCuda;
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
