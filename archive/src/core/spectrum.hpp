#pragma once

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

double EvalSpectrumAmplitude(const std::vector<double> &wavelengths, const std::vector<double> &amplitudes, double lambda);

double AverageSpectrumSamples(const std::vector<double> &wavelengths, const std::vector<double> &amplitudes,
                              double wavelength_start, double wavelength_end);

dvec3 SpectrumToRgb(const std::vector<double> &wavelengths, const std::vector<double> &amplitudes);

NAMESPACE_END(raytracer)