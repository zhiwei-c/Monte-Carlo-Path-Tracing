#ifndef CSRT__RAY_TRACER_HPP
#define CSRT__RAY_TRACER_HPP

#include <string>

#include "parser/parser.hpp"
#include "renderer/renderer.hpp"

namespace csrt
{

class RayTracer
{
public:
    RayTracer(const Config &config);
    ~RayTracer();

    void Draw(const std::string &output_filename) const;

#ifdef ENABLE_VIEWER
    void Preview(int argc, char **argv, const std::string &output_filename);
#endif

private:
    BackendType backend_type_;
    Scene *scene_;
    Renderer *renderer_;
    float *frame_;
#ifdef ENABLE_VIEWER
    float *accum_;
#endif
    int width_;
    int height_;
};

} // namespace csrt

#endif