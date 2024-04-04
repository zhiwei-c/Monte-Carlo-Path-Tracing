#ifndef CSRT__UTILS__MISC_HPP
#define CSRT__UTILS__MISC_HPP

#include <exception>
#include <string>

namespace csrt
{

#define SAFE_DELETE_ELEMENT(x)                                                 \
    {                                                                          \
        if (x)                                                                 \
        {                                                                      \
            delete (x);                                                        \
        }                                                                      \
        x = nullptr;                                                           \
    }

#define SAFE_DELETE_ARRAY(x)                                                   \
    {                                                                          \
        if (x)                                                                 \
        {                                                                      \
            delete[] (x);                                                      \
        }                                                                      \
        x = nullptr;                                                           \
    }

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

class MyException : public std::exception
{
public:
    MyException(const std::string &info) : info_(info) {}

    virtual char const *what() const throw() { return info_.c_str(); }

private:
    std::string info_;
};

} // namespace csrt

#endif