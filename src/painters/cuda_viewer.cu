#include "painters.cuh"

#ifdef ENABLE_VIEWER

#include <chrono>
#include <gl/freeglut.h>

#include "../utils/math.cuh"
#include "../utils/image_io.cuh"

namespace
{
    char text[256];
    int raw_width, raw_height;
    int window_width, window_height;
    uint32_t index_frame = 0, last_time = 0, current_time = 0;
    float *frame_raw = nullptr, *frame_device = nullptr, *frame_host = nullptr, *frame_screen = nullptr;
    double fps = 0.0, count_frame_cyc = 0.0, aspect;
    dim3 num_blocks, threads_per_block;
    std::string filename;
    Camera *camera = nullptr;
    Integrator *integrator = nullptr;

    __global__ void DispatchRays(Camera *camera, Integrator *integrator, uint32_t index_frame,
                                 float *frame_raw, float *frame_device)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < camera->width() && j < camera->height())
        {
            const uint32_t pixel_index = j * camera->width() + i;
            uint32_t seed = Tea(pixel_index, index_frame, 4);

            const float u = GetVanDerCorputSequence(index_frame + 1, 2),
                        v = GetVanDerCorputSequence(index_frame + 1, 3),
                        x = 2.0f * (i + u) / camera->width() - 1.0f,
                        y = 2.0f * (j + v) / camera->height() - 1.0f;
            const Vec3 look_dir = Normalize(camera->front() + x * camera->view_dx() +
                                            y * camera->view_dy());
            Vec3 color = integrator->GenerateRay(camera->eye(), look_dir, &seed);

            float color_temp;
            for (int c = 0; c < 3; ++c)
            {
                frame_raw[pixel_index * 3 + c] = (index_frame * frame_raw[pixel_index * 3 + c] +
                                                  color[c]) /
                                                 (index_frame + 1);
                color_temp = fminf(frame_raw[pixel_index * 3 + c], 1.0f);

                if (color_temp <= 0.0031308f)
                    frame_device[pixel_index * 3 + c] = 12.92f * color_temp;
                else
                    frame_device[pixel_index * 3 + c] = 1.055f * powf(color_temp, 1.0f / 2.4f) -
                                                        0.055f;
            }
        }
    }

    void Render()
    {
        // clear the viewport
        glClear(GL_COLOR_BUFFER_BIT);

        // 绘制图形
        DispatchRays<<<num_blocks, threads_per_block>>>(camera, integrator, index_frame, frame_raw,
                                                        frame_device);
        CheckCudaErrors(cudaGetLastError());
        CheckCudaErrors(cudaDeviceSynchronize());
        CheckCudaErrors(cudaMemcpy(frame_host, frame_device,
                                   3 * raw_width * raw_height * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        image_io::Resize(frame_host, raw_width, raw_height, 0,
                         frame_screen, window_width, window_height, 0, 3);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0, window_width, 0, window_height);

        glRasterPos2i(0, 0);
        glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, (GLvoid *)frame_screen);

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
            std::vector<float> data(3 * raw_width * raw_height);
            uint32_t offset_src, offset_dst;
            for (int j = 0; j < raw_height; ++j)
            {
                for (int i = 0; i < raw_width; ++i)
                {
                    offset_src = (j * raw_width + i) * 3;
                    offset_dst = ((raw_height - 1 - j) * raw_width + i) * 3;
                    for (int k = 0; k < 3; ++k)
                        data[offset_dst + k] = frame_host[offset_src + k];
                }
            }
            image_io::Write(raw_width, raw_height, data.data(), filename);
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
        if (window_width != glutGet(GLUT_WINDOW_WIDTH))
        {
            window_width = glutGet(GLUT_WINDOW_WIDTH);
            window_height = static_cast<int>(window_width / aspect);
        }
        else
        {
            window_height = glutGet(GLUT_WINDOW_HEIGHT);
            window_width = static_cast<int>(window_height * aspect);
        }

        glutReshapeWindow(window_width, window_height);

        SAFE_DELETE_ARRAY(frame_screen);
        frame_screen = new float[3 * window_width * window_height];

        // tell glut to redraw
        glutPostRedisplay();
    }
} // namespace

CudaViewer::CudaViewer(int argc, char **argv, BvhBuilder::Type bvh_type, const SceneInfo &info)
    : CudaPainter(bvh_type, info)
{
    ::camera = camera_;
    ::integrator = integrator_;
    ::raw_width = ::window_width = info.camera.width();
    ::raw_height = ::window_height = info.camera.height();
    ::aspect = static_cast<double>(info.camera.width()) / info.camera.height();
    ::threads_per_block = {8, 8, 1};
    ::num_blocks = {static_cast<unsigned int>(camera_->width() / 8 + 1),
                    static_cast<unsigned int>(camera_->height() / 8 + 1), 1};

    uint32_t num_component = 3 * window_width * window_height;
    ::frame_host = new float[num_component];
    CheckCudaErrors(cudaMallocManaged((void **)&::frame_raw, num_component * sizeof(float)));
    CheckCudaErrors(cudaMallocManaged((void **)&::frame_device, num_component * sizeof(float)));

    // Initialize GLUT
    glutInit(&argc, argv);

    // Create a window
    glutInitWindowSize(::window_width, ::window_height);
    glutInitWindowPosition(20, 20);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutCreateWindow("Ray Tracing");

    // Register callback functions
    glutDisplayFunc(::Render);
    glutKeyboardFunc(::PressCharkey);
    glutSpecialFunc(::PressSpeckey);
    glutReshapeFunc(::Reshape);

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

    // set the background color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

CudaViewer::~CudaViewer()
{
    SAFE_DELETE_ARRAY(::frame_host);

    CheckCudaErrors(cudaGetLastError());
    CheckCudaErrors(cudaFree(::frame_device));
}

void CudaViewer::Draw(const std::string &filename)
{
    ::filename = filename;
    ::last_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();

    // call main loop
    glutMainLoop();
}

#endif