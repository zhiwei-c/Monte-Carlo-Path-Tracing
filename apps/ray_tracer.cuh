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

private:
    csrt::BackendType backend_type_;
    csrt::Scene *scene_;
    csrt::Renderer *renderer_;
    float *frame_;
    int width_;
    int height_;
};