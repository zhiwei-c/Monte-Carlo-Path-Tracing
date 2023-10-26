#pragma once

#include <cstdint>
#include <cstdio>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#define NAMESPACE_BEGIN(name) namespace name {
#define NAMESPACE_END(name) }

#ifndef QUALIFIER_DEVICE
#ifdef ENABLE_CUDA
#define QUALIFIER_DEVICE __device__ __host__
#else
#define QUALIFIER_DEVICE
#endif
#endif