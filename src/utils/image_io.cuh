#pragma once

#include <string>
#include <vector>

namespace image_io
{
    void Write(int width, int height, float *frame_buffer, const std::string &filename);
    void Read(const std::string &filename, float gamma, int *width, int *height, int *channel,
              std::vector<float> *data, int* width_max);
} // namespace image_io
