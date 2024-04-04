#ifndef CSRT__PARSER__SPECTRUM_HPP
#define CSRT__PARSER__SPECTRUM_HPP

#include <vector>

#include "../tensor.hpp"

namespace csrt
{

float EvalSpectrumAmplitude(const std::vector<float> &wavelengths,
                            const std::vector<float> &amplitudes,
                            const float lambda);
float AverageSpectrumSamples(const std::vector<float> &wavelengths,
                             const std::vector<float> &amplitudes,
                             const float wavelength_start,
                             const float wavelength_end);
Vec3 SpectrumToRgb(const std::vector<float> &wavelengths,
                   const std::vector<float> &amplitudes);

} // namespace csrt

#endif