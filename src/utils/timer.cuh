#pragma once

#include <string>
#include <chrono>

class Timer
{
public:
    Timer();

    void Reset();
    void PrintTimePassed(const std::string &work_name = "");
    void PrintProgress(double progress);

private:
    std::chrono::steady_clock::time_point time_begin_; 
    std::chrono::steady_clock::time_point time_current_; 
};
