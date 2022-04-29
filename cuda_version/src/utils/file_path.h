#pragma once

#include <string>

inline std::string ConvertBackSlash(std::string path)
{
    if (path.find_last_of("\\") != std::string::npos)
    {
        for (size_t i = 0; i < path.length(); i++)
        {
            if (path[i] == '\\')
                path[i] = '/';
        }
    }

    return path;
}

inline std::string GetDirectory(const std::string& path)
{
    if (path.find_last_of("/\\") != std::string::npos)
        return path.substr(0, path.find_last_of("/\\")) + "/";
    else
        return "";
}

inline std::string GetSuffix(const std::string& path)
{
    if (path.find_last_of(".") != std::string::npos)
        return path.substr(path.find_last_of(".") + 1);
    else
        return "";
}

inline std::string ChangeSuffix(const std::string& path, const std::string& new_suffix)
{
    if (path.find_last_of(".") != std::string::npos)
        return path.substr(0, path.find_last_of(".")) + "." + new_suffix;
    else
        return path + "." + new_suffix;
}


