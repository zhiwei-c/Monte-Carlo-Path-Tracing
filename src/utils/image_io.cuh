#pragma once

#include <string>
#include <vector>

namespace image_io
{
    void Write(int width, int height, float *frame_buffer, const std::string &filename);
    void Read(const std::string &filename, float gamma, int *width, int *height, int *channel,
              std::vector<float> *data, int *width_max);
    void Resize(const float *input_pixels, int input_w, int input_h, int input_stride_in_bytes,
                float *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                int num_channels);
} // namespace image_io
