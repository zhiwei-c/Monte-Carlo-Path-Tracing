#pragma once

#include "../global.hpp"

#include <string>

#include <ctime>

NAMESPACE_BEGIN(raytracer)

//计时器类
class Timer
{
public:
    Timer();

    void Reset();
    void PrintTimePassed(const std::string& work_name = "");
    void PrintProgress(double progress);

private:
    time_t start_; //开始计时的时刻
};

NAMESPACE_END(raytracer)