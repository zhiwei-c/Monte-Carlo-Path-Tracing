#pragma once

#include <cstdint>
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_CUDA
#define QUALIFIER_D_H __device__ __host__
#define QUALIFIER_HOST __host__
#else
#define QUALIFIER_D_H
#define QUALIFIER_HOST
#endif