#include "ray_tracer.cuh"

RayTracer::RayTracer(const csrt::Config &config)
    : backend_type_(config.backend_type), width_(config.camera.width),
      height_(config.camera.height)
{
    try
    {
        scene_ = new csrt::Scene(config.backend_type);
        for (size_t i = 0; i < config.instances.size(); ++i)
            scene_->AddInstance(config.instances[i]);
        scene_->Commit();

        renderer_ = new csrt::Renderer(config.backend_type);

        for (size_t i = 0; i < config.textures.size(); ++i)
            renderer_->AddTexture(config.textures[i]);

        for (size_t i = 0; i < config.bsdfs.size(); ++i)
            renderer_->AddBsdf(config.bsdfs[i]);

        std::vector<uint32_t> map_id_instance_bsdf;
        for (size_t i = 0; i < config.instances.size(); ++i)
            map_id_instance_bsdf.push_back(config.instances[i].id_bsdf);

        renderer_->AddSceneInfo(scene_->GetInstances(),
                                scene_->GetPdfAreaList(), map_id_instance_bsdf,
                                scene_->GetTlas());

        for (size_t i = 0; i < config.emitters.size(); ++i)
            renderer_->AddEmitter(config.emitters[i]);

        std::vector<uint32_t> map_id_area_light_instance;
        std::vector<float> list_area_light_weight;
        for (size_t i = 0; i < config.instances.size(); ++i)
        {
            const csrt::BSDF::Info info_bsdf =
                config.bsdfs[config.instances[i].id_bsdf];
            if (info_bsdf.type == csrt::BSDF::Type::kAreaLight)
            {
                map_id_area_light_instance.push_back(i);
                list_area_light_weight.push_back(info_bsdf.area_light.weight);
            }
        }
        renderer_->SetAreaLightInfo(map_id_area_light_instance,
                                    list_area_light_weight);

        renderer_->SetCamera(config.camera);
        renderer_->SetIntegrator(config.integrator);
        renderer_->Commit();

        uint32_t num_element = config.camera.width * config.camera.height * 3;
        frame_ = csrt::MallocArray<float>(config.backend_type, num_element);
        frame_[0] = 0;
        frame_[num_element - 1] = 0;
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

RayTracer::~RayTracer()
{
    csrt::DeleteArray(backend_type_, frame_);

    delete renderer_;
    delete scene_;
}

void RayTracer::Draw(const std::string &output_filename) const
{
    try
    {
        renderer_->Draw(frame_);
        csrt::image_io::Write(width_, height_, frame_, output_filename);
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}