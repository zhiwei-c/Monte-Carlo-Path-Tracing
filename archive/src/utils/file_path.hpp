#pragma once

#include "../global.hpp"

NAMESPACE_BEGIN(raytracer)

///\brief 获取路径中的目录
inline std::string GetDirectory(const std::string &path)
{
    if (path.find_last_of("/\\") != std::string::npos)
        return path.substr(0, path.find_last_of("/\\")) + "/";
    else
        return "";
}

///\brief 获取路径中的文件后缀
inline std::string GetSuffix(const std::string &path)
{
    if (path.find_last_of(".") != std::string::npos)
        return path.substr(path.find_last_of(".") + 1);
    else
        return "";
}

///\brief 更改路径中的文件后缀
inline std::string ChangeSuffix(const std::string &path, const std::string &new_suffix)
{
    if (path.find_last_of(".") != std::string::npos)
        return path.substr(0, path.find_last_of(".")) + "." + new_suffix;
    else
        return path + "." + new_suffix;
}
NAMESPACE_END(raytracer)