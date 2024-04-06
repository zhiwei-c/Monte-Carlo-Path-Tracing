#ifndef CSRT__UTILS__MEMORY_HPP
#define CSRT__UTILS__MEMORY_HPP

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cstdint>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace csrt
{

enum class BackendType
{
    kCpu,
#ifdef ENABLE_CUDA
    kCuda
#endif
};

template <typename T>
T *MallocElement(const BackendType &backend_type)
{
#ifdef ENABLE_CUDA
    if (backend_type == BackendType::kCpu)
    {
#endif
        return new T;
#ifdef ENABLE_CUDA
    }
    else
    {
        T *data = nullptr;
        cudaError_t ret = cudaMallocManaged(&data, sizeof(T));
        switch (ret)
        {
        case 0:
        {
            break;
        }
        case 2:
        {
            cudaDeviceReset();
            throw MyException(
                "cannot allocate enough GPU memory or other resources.");
            break;
        }
        default:
        {
            cudaDeviceReset();
            std::ostringstream oss;
            oss << "CUDA error : \"" << ret << "\". when malloc memory.";
            throw MyException(oss.str());
            break;
        }
        }
        return data;
    }
#endif
}

template <typename T>
T *MallocArray(const BackendType &backend_type, const uint64_t num)
{
#ifdef ENABLE_CUDA
    if (backend_type == BackendType::kCpu)
    {
#endif
        return new T[num];
#ifdef ENABLE_CUDA
    }
    else
    {
        T *data = nullptr;
        cudaError_t ret = cudaMallocManaged(&data, num * sizeof(T));
        switch (ret)
        {
        case 0:
        {
            break;
        }
        case 2:
        {
            cudaDeviceReset();
            throw MyException(
                "cannot allocate enough GPU memory or other resources.");
            break;
        }
        default:
        {
            cudaDeviceReset();
            std::ostringstream oss;
            oss << "CUDA error : \"" << ret << "\". when malloc memory.";
            throw MyException(oss.str());
            break;
        }
        }
        return data;
    }
#endif
}

template <typename T>
T *MallocArray(const BackendType &backend_type, const std::vector<T> &src)
{
    uint64_t num = src.size();

#ifdef ENABLE_CUDA
    if (backend_type == BackendType::kCpu)
    {
#endif
        T *dest = new T[num];
        std::copy(src.begin(), src.end(), dest);
        return dest;
#ifdef ENABLE_CUDA
    }
    else
    {
        T *dest = nullptr;
        cudaError_t ret = cudaMallocManaged(&dest, num * sizeof(T));
        switch (ret)
        {
        case 0:
        {
            break;
        }
        case 2:
        {
            cudaDeviceReset();
            throw MyException(
                "cannot allocate enough GPU memory or other resources.");
            break;
        }
        default:
        {
            cudaDeviceReset();
            std::ostringstream oss;
            oss << "CUDA error : \"" << ret << "\". when malloc memory.";
            throw MyException(oss.str());
            break;
        }
        }

        ret = cudaMemcpy(dest, src.data(), num * sizeof(T),
                         cudaMemcpyHostToDevice);
        if (ret)
        {
            cudaDeviceReset();
            std::ostringstream oss;
            oss << "CUDA error : \"" << ret << "\". when copy memory.";
            throw MyException(oss.str());
        }

        return dest;
    }
#endif
}

template <typename T>
void CopyArray(const BackendType &backend_type, T *&dest, const T *src,
               const uint64_t num)
{
#ifdef ENABLE_CUDA
    if (backend_type == BackendType::kCpu)
    {
#endif
        std::copy(src, src + num, dest);
#ifdef ENABLE_CUDA
    }
    else
    {
        cudaError_t ret =
            cudaMemcpy(dest, src, num * sizeof(T), cudaMemcpyDefault);
        if (ret)
        {
            cudaDeviceReset();
            std::ostringstream oss;
            oss << "CUDA error : \"" << ret << "\". when copy memory.";
            throw MyException(oss.str());
        }
    }
#endif
}

template <typename T>
void DeleteElement(const BackendType &backend_type, T *&data)
{
    if (data == nullptr)
        return;

#ifdef ENABLE_CUDA
    if (backend_type == BackendType::kCpu)
    {
#endif
        delete data;
        data = nullptr;
#ifdef ENABLE_CUDA
    }
    else
    {
        cudaError_t ret = cudaFree(data);
        if (ret)
        {
            cudaDeviceReset();
            std::ostringstream oss;
            oss << "CUDA error : \"" << ret << "\" when delete memory.";
            throw MyException(oss.str());
        }
        data = nullptr;
    }
#endif
}

template <typename T>
void DeleteArray(const BackendType &backend_type, T *&data)
{
    if (data == nullptr)
        return;

#ifdef ENABLE_CUDA
    if (backend_type == BackendType::kCpu)
    {
#endif
        delete[] data;
        data = nullptr;
#ifdef ENABLE_CUDA
    }
    else
    {
        cudaError_t ret = cudaFree(data);
        if (ret)
        {
            cudaDeviceReset();
            std::ostringstream oss;
            oss << "CUDA error : \"" << ret << "\" when delete memory.";
            throw MyException(oss.str());
        }
        data = nullptr;
    }
#endif
}

} // namespace csrt

#endif