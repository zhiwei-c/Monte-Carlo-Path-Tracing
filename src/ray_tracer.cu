#include "csrt/ray_tracer.cuh"

#ifdef ENABLE_VIEWER

#include <chrono>
#include <gl/freeglut.h>

namespace
{

char text[256];
int screen_width, screen_height;
uint32_t index_frame = 0, last_time = 0, current_time = 0;
float *accum = nullptr, *frame = nullptr;
double fps = 0.0, count_frame_cyc = 0.0, aspect;
std::string filename;
csrt::Renderer *renderer;

void Render()
{
    // clear the viewport
    glClear(GL_COLOR_BUFFER_BIT);

    // 绘制图形
    try
    {
        renderer->Draw(index_frame, accum, frame);
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "[error] %s.\n", e.what());
        cudaDeviceReset();
        exit(1);
    }

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, screen_width, 0, screen_height);

    glRasterPos2i(0, 0);
    glDrawPixels(screen_width, screen_height, GL_RGB, GL_FLOAT,
                 (GLvoid *)frame);

    // 计算 FPS
    current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    if (current_time - last_time > 1000)
    {
        fps = 1000.0 * count_frame_cyc / (current_time - last_time);
        count_frame_cyc = 0.0;
        last_time = current_time;
        snprintf(text, sizeof(text), "FPS %.3lf ", fps);
    }
    glRasterPos2f(0, 0);
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    const unsigned char *ustr = reinterpret_cast<const unsigned char *>(text);
    glutBitmapString(GLUT_BITMAP_HELVETICA_18, ustr);

    // swap buffers
    glutSwapBuffers();

    ++index_frame;
    count_frame_cyc += 1.0;

    // tell glut to redraw
    glutPostRedisplay();
}

void PressCharkey(unsigned char key, int x, int y)
{
    switch (key)
    {
    case 's':
    case 'S':
    {
        std::vector<float> data(3 * screen_width * screen_height);
        uint32_t offset_src, offset_dst;
        for (int j = 0; j < screen_height; ++j)
        {
            for (int i = 0; i < screen_width; ++i)
            {
                offset_src = (j * screen_width + i) * 3;
                offset_dst = ((screen_height - 1 - j) * screen_width + i) * 3;
                for (int k = 0; k < 3; ++k)
                    data[offset_dst + k] = frame[offset_src + k];
            }
        }
        csrt::image_io::Write(screen_width, screen_height, data.data(),
                              filename);
        break;
    }
    case 27:
        glutLeaveMainLoop();
        break;
    default:
        break;
    }

    // tell glut to redraw
    glutPostRedisplay();
}

void PressSpeckey(int key, int x, int y)
{
    switch (key)
    {
    case GLUT_KEY_PAGE_UP:
    default:
        break;
    }

    // tell glut to redraw
    glutPostRedisplay();
}

void Reshape(int x, int y)
{
    // do what you want when window size changes
    glutReshapeWindow(screen_width, screen_height);

    // tell glut to redraw
    glutPostRedisplay();
}

} // namespace

#endif

namespace csrt
{

RayTracer::RayTracer(const csrt::Config &config)
    : backend_type_(config.backend_type), width_(config.camera.width),
      height_(config.camera.height)
#ifdef ENABLE_VIEWER
    , accum_(nullptr)
#endif
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
    }
    catch (const std::exception &e)
    {
        throw e;
    }
}

RayTracer::~RayTracer()
{
    csrt::DeleteArray(backend_type_, frame_);
#ifdef ENABLE_VIEWER
    csrt::DeleteArray(backend_type_, accum_);
#endif

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

#ifdef ENABLE_VIEWER
void RayTracer::Preview(int argc, char **argv,
                        const std::string &output_filename)
{
    // Initialize GLUT
    glutInit(&argc, argv);

    // Create a window
    glutInitWindowSize(width_, height_);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutCreateWindow("Ray Tracing");

    // Register callback functions
    glutDisplayFunc(::Render);
    glutKeyboardFunc(::PressCharkey);
    glutSpecialFunc(::PressSpeckey);
    glutReshapeFunc(::Reshape);

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // set the background color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    ::screen_width = ::screen_width = width_;
    ::screen_height = ::screen_height = height_;
    ::aspect = static_cast<double>(width_) / height_;
    ::renderer = renderer_;
    ::filename = output_filename;
    ::last_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

    accum_ = csrt::MallocArray<float>(backend_type_, width_ * height_ * 3);
    ::accum = accum_;
    ::frame = frame_;

    // call main loop
    glutMainLoop();
}
#endif

} // namespace csrt