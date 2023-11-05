#pragma once

#include <string>
#include <vector>

namespace rt
{

namespace image_io
{
    void Write(const int width, const int height, const float *frame_buffer,
               const std::string &filename);
    void Read(const std::string &filename, const float gamma, const int *width_max, int *width,
              int *height, int *channel, std::vector<float> *data);
    void Resize(const float *input_pixels, int input_w, int input_h, int input_stride_in_bytes,
                float *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                int num_channels);
} // namespace image_io

} // namespace rt
