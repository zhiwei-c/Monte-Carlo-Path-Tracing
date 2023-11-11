#include <cstdio>

#include "../test_rt/model_loader.cuh"
#include "ray_tracer.cuh"

using namespace rt;

struct Config
{
    BackendType backend_type;
    Camera::Info camera;
    Integrator::Info integrator;
    std::vector<Texture::Info> textures;
    std::vector<Bsdf::Info> bsdfs;
    std::vector<Instance::Info> instances;
};

Config SetupConfig(const BackendType backend_type)
{
    Config config;

    config.backend_type = backend_type;

    config.camera.spp = 64;
    config.integrator.pdf_rr = 0.99;

    //
    // texture
    //
    // 0 - White
    config.textures.push_back(
        Texture::Info::CreateConstant({0.725f, 0.71f, 0.68f}));
    // 1 - Green
    config.textures.push_back(
        Texture::Info::CreateConstant({0.14f, 0.45f, 0.091f}));
    // 2 - Red
    config.textures.push_back(
        Texture::Info::CreateConstant({0.63f, 0.065f, 0.05f}));
    // 3 - Light radiance
    config.textures.push_back(
        Texture::Info::CreateConstant({17.0f, 12.0f, 4.0f}));
    // 4 value 0.05
    config.textures.push_back(Texture::Info::CreateConstant({0.05f}));
    // 5 value 1
    config.textures.push_back(Texture::Info::CreateConstant({1.0f}));
    // 6 value 0.5
    config.textures.push_back(Texture::Info::CreateConstant({0.5f}));
    // 7 Green 2
    config.textures.push_back(
        Texture::Info::CreateConstant({0.240f, 0.771f, 0.361f}));

    //
    // BSDF
    //
    // 0 White
    config.bsdfs.push_back(
        Bsdf::Info::CreateDiffuse(0, true, kInvalidId, kInvalidId));
    // 1 Green
    config.bsdfs.push_back(
        Bsdf::Info::CreateDiffuse(1, true, kInvalidId, kInvalidId));
    // 2 Red
    config.bsdfs.push_back(
        Bsdf::Info::CreateDiffuse(2, true, kInvalidId, kInvalidId));
    // 3 Light
    config.bsdfs.push_back(
        Bsdf::Info::CreateAreaLight(3, 1.0f, false, kInvalidId, kInvalidId));
    // 4 conductor
    config.bsdfs.push_back(Bsdf::Info::CreateConductor(
        4, 6, 5, {1.65394, 0.87850, 0.52012}, {9.20430, 6.25621, 4.82675}, true,
        kInvalidId, kInvalidId));
    // 5 dielectric
    config.bsdfs.push_back(Bsdf::Info::CreateDielectric(
        false, 4, 4, 5, 5, 1.5f, false, kInvalidId, kInvalidId));
    // 6 thin dielectric
    config.bsdfs.push_back(Bsdf::Info::CreateDielectric(
        true, 4, 4, 5, 5, 1.5f, false, kInvalidId, kInvalidId));
    // 7 plastic
    config.bsdfs.push_back(
        Bsdf::Info::CreatePlastic(1.9, 4, 7, 5, true, kInvalidId, kInvalidId));

    //
    // Instance
    //
    // 0 Floor
    config.instances.push_back(Instance::Info::CreateRectangle(
        {{0, 1, 0, 0}, {0, 0, 2, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}}, 0));
    // 1 Ceiling
    config.instances.push_back(Instance::Info::CreateRectangle(
        {{-1, 0, 0, 0}, {0, 0, -2, 2}, {0, -1, 0, 0}, {0, 0, 0, 1}}, 0));
    // 2 BackWall
    config.instances.push_back(Instance::Info::CreateRectangle(
        {{0, 1, 0, 0}, {1, 0, 0, 1}, {0, 0, -2, -1}, {0, 0, 0, 1}}, 0));
    // 3 RightWall
    config.instances.push_back(Instance::Info::CreateRectangle(
        {{0, 0, 2, 1}, {1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 1}}, 1));
    // 4 LeftWall
    config.instances.push_back(Instance::Info::CreateRectangle(
        {{0, 0, -2, -1}, {1, 0, 0, 1}, {0, -1, 0, 0}, {0, 0, 0, 1}}, 2));
    // 5 Light
    config.instances.push_back(
        Instance::Info::CreateRectangle({{0.235, 0, 0, -0.005},
                                         {0, 0, -0.0893, 1.98},
                                         {0, 0.19, 0, -0.03},
                                         {0, 0, 0, 1}},
                                        3));
    // // 6 happy Buddha
    // Instance::Info info;
    // info.type = Instance::Type::kMeshes;
    // info.id_bsdf = 4;
    // info.meshes = model_loader::Load(
    //     GetDirectory(__FILE__) + "../test_rt/happy_vrip_2.ply", true, false);
    // info.meshes.to_world = Scale({6.0f, 6.0f, 6.0f});
    // info.meshes.to_world =
    //     Mul(Translate({0.0327015072f, -0.299091011f, 0.0402149931f}),
    //         info.meshes.to_world);
    // info.meshes.to_world = Mul(Translate({-0.5f, 0, 0}),
    // info.meshes.to_world); config.instances.push_back(info);

    // // 7 Bunny
    // info = {};
    // info.type = Instance::Type::kMeshes;
    // info.id_bsdf = 5;
    // info.meshes = model_loader::Load(
    //     GetDirectory(__FILE__) + "../test_rt/bun_zipper_1.ply", true, false);
    // info.meshes.to_world = Scale({4.0f, 4.0f, 4.0f});
    // info.meshes.to_world =
    //     Mul(Translate({0.0672739968f, -0.133556396f, 0.00636819750f}),
    //         info.meshes.to_world);
    // info.meshes.to_world = Mul(Translate({0.5f, 0, 0}),
    // info.meshes.to_world); config.instances.push_back(info);

    // 8 Sphere
    config.instances.push_back(
        Instance::Info::CreateSphere(0.5f, {0.0f, 0.5f, -0.1f}, {}, 4));

    // // 6  Tall Box
    // config.instances.push_back(
    //     Instance::Info::CreateCube({{0.286776, 0.098229, 0, -0.335439},
    //                                 {0, 0, -0.6, 0.6},
    //                                 {-0.0997984, 0.282266, 0, -0.291415},
    //                                 {0, 0, 0, 1}},
    //                                0));
    // // 7 Short Box
    // config.instances.push_back(
    //     Instance::Info::CreateCube({{0.0851643, 0.289542, 0, 0.328631},
    //                                 {0, 0, -0.3, 0.3},
    //                                 {-0.284951, 0.0865363, 0, 0.374592},
    //                                 {0, 0, 0, 1}},
    //                                0));
    // // 8 Sphere
    // config.instances.push_back(Instance::Info::CreateSphere(
    //     0.15f, {0.328631014f, 0.75f, 0.374592006f}, {}, 0));

    return config;
}

void Test(const BackendType backend_type, const std::string &output_filename)
{
    Config config = SetupConfig(backend_type);

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

int main()
{
    Test(BackendType::kCpu, "cpu.png");
    Test(BackendType::kCuda, "cuda.png");
    return 0;
}
