#pragma once

#include <string>

#include "parser.cuh"
#include "renderer.cuh"

class RayTracer
{
public:
    RayTracer(const csrt::Config &config);
    ~RayTracer();

    void Draw(const std::string &output_filename) const;

#ifdef ENABLE_VIEWER
    void Preview(int argc, char **argv, const std::string &output_filename);
#endif

private:
    csrt::BackendType backend_type_;
    csrt::Scene *scene_;
    csrt::Renderer *renderer_;
    float *frame_;
#ifdef ENABLE_VIEWER
    float *accum_;
#endif
    int width_;
    int height_;
};