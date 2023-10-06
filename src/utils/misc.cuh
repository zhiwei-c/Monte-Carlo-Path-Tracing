#pragma once

#include <string>
#include <cstdio>

#include "../global.cuh"

#define SAFE_DELETE_ELEMENT(x) \
    {                          \
        if (x)                 \
        {                      \
            delete (x);        \
        }                      \
        x = nullptr;           \
    }

#define SAFE_DELETE_ARRAY(x) \
    {                        \
        if (x)               \
        {                    \
            delete[] (x);    \
        }                    \
        x = nullptr;         \
    }

#define CheckCudaErrors(val) CheckCuda((val), #val, __FILE__, __LINE__)
#define PrintExcuError() PrintError(__FUNCTION__, __FILE__, __LINE__)

inline std::string GetSuffix(const std::string &filename)
{
    if (filename.find_last_of(".") != std::string::npos)
        return filename.substr(filename.find_last_of(".") + 1);
    else
        return "";
}

inline std::string GetDirectory(const std::string &path)
{
    if (path.find_last_of("/\\") != std::string::npos)
        return path.substr(0, path.find_last_of("/\\")) + "/";
    else
        return "";
}

constexpr uint64_t Hash(const char *str)
{
    return (*str ? Hash(str + 1) * 256 : 0) + static_cast<uint64_t>(*str);
}

constexpr uint64_t operator"" _hash(const char *str, uint64_t)
{
    return Hash(str);
}

#ifdef ENABLE_CUDA

inline void CheckCuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error = %llu at %s : %d inside function '%s' \n", static_cast<uint64_t>(result), file, line, func);
        cudaDeviceReset();
        exit(99);
    }
}

inline void PrintError(char const *const func, const char *const file, int const line)
{
    fprintf(stderr, "error at %s : %d inside function '%s' \n", file, line, func);
    cudaDeviceReset();
    exit(1);
}

#endif