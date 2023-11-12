#include <cstdio>

#include "ray_tracer.cuh"

using namespace csrt;

void Test(const Config &config, const std::string &output_filename)
{
    Scene *scene = new Scene(config.backend_type);
    for (size_t i = 0; i < config.instances.size(); ++i)
        scene->AddInstance(config.instances[i]);
    scene->Commit();

    Renderer *renderer = new Renderer(config.backend_type);
    for (size_t i = 0; i < config.textures.size(); ++i)
        renderer->AddTexture(config.textures[i]);
    for (size_t i = 0; i < config.bsdfs.size(); ++i)
        renderer->AddBsdf(config.bsdfs[i]);

    std::vector<uint32_t> map_id_instance_bsdf;
    for (size_t i = 0; i < config.instances.size(); ++i)
        map_id_instance_bsdf.push_back(config.instances[i].id_bsdf);

    renderer->AddSceneInfo(map_id_instance_bsdf, scene->GetInstances(),
                           scene->GetPdfAreaList(), scene->GetTlas());

    std::vector<uint32_t> map_id_area_light_instance;
    std::vector<float> list_area_light_weight;
    for (size_t i = 0; i < config.instances.size(); ++i)
    {
        const Bsdf::Info info_bsdf = config.bsdfs[config.instances[i].id_bsdf];
        if (info_bsdf.type == Bsdf::Type::kAreaLight)
        {
            map_id_area_light_instance.push_back(i);
            list_area_light_weight.push_back(info_bsdf.area_light.weight);
        }
    }
    renderer->SetAreaLightInfo(map_id_area_light_instance,
                               list_area_light_weight);

    renderer->SetCamera(config.camera);
    renderer->SetIntegrator(config.integrator);
    renderer->Commit();

    float *frame = MallocArray<float>(
        config.backend_type, config.camera.width * config.camera.height * 3);

    renderer->Draw(frame);

    image_io::Write(config.camera.width, config.camera.height, frame,
                    output_filename);

    DeleteArray(config.backend_type, frame);

    delete renderer;
    delete scene;
}

struct Param
{
    BackendType type;
    std::string input;
    std::string output;

    Param() : type(BackendType::kCpu), input(""), output("result.png") {}
};

Param ParseParam(int argc, char **argv);

int main(int argc, char **argv)
{
    Param param = ParseParam(argc, argv);
    Config confg = LoadConfig(param.input);
    confg.backend_type = param.type;
    Test(confg, param.output);
    return 0;
}

Param ParseParam(int argc, char **argv)
{
    Param param;
    for (int i = 0; i < argc; ++i)
    {
        if (argv[i] == std::string("--cpu") || argv[i] == std::string("-c"))
        {
            param.type = BackendType::kCpu;
        }
        else if (argv[i] == std::string("--gpu") ||
                 argv[i] == std::string("--g"))
        {
            param.type = BackendType::kCuda;
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

    if (GetSuffix(param.output) != "png")
    {
        fprintf(
            stderr,
            "[warning] only support png output, ignore output format \"%s\".",
            GetSuffix(param.output).c_str());
        param.output =
            param.output.substr(0, param.output.find_last_of(".")) + ".png";
    }

    return param;
}
