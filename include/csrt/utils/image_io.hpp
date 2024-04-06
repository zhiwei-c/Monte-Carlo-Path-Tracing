#ifndef CSRT__UTILS__IMAGE_IO_HPP
#define CSRT__UTILS__IMAGE_IO_HPP

#include <string>
#include <vector>

namespace csrt
{

namespace image_io
{
    void Write(const float *data, const int width, const int height,
               std::string filename);
    float *Read(const std::string &filename, const float gamma,
                const int *width_max, int *width, int *height, int *channel);
    void Resize(const float *input_pixels, int input_w, int input_h,
                int input_stride_in_bytes, float *output_pixels, int output_w,
                int output_h, int output_stride_in_bytes, int num_channels);
} // namespace image_io

} // namespace csrt

#endif