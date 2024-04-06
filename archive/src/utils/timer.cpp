#include "timer.hpp"

#include <iostream>
#include <iomanip>

NAMESPACE_BEGIN(raytracer)

Timer::Timer()
{
    time(&start_);
}

void Timer::Reset()
{
    time(&start_);
}

void Timer::PrintTimePassed(const std::string &work_name)
{
    time_t now;
    time(&now);
    auto diff = difftime(now, start_);
    int hours = static_cast<int>(diff / 3600),
        mins = static_cast<int>(diff / 60) - hours * 60,
        secs = static_cast<int>(diff) - hours * 3600 - mins * 60;
    std::cerr << "[info] \"" << work_name << "\" Finished.\n"
              << "\tIt takes " << hours << " h, " << mins << " m, " << secs << " s.\n";
}

void Timer::PrintProgress(double progress)
{
    time_t now;
    time(&now);

    auto diff = difftime(now, start_);
    int hours = static_cast<int>(diff / 3600),
        mins = static_cast<int>(diff / 60) - hours * 60,
        secs = static_cast<int>(diff) - hours * 3600 - mins * 60;

    auto diff_predict = (progress > 1e-4) ? diff / progress : 0;
    int hours_predict = static_cast<int>(diff_predict / 3600),
        mins_predict = static_cast<int>(diff_predict / 60) - hours_predict * 60,
        secs_predict = static_cast<int>(diff_predict) - hours_predict * 3600 - mins_predict * 60;

    std::cerr << "\r" << progress * 100 << "%"
              << " [" << hours << ":" << mins << ":" << secs << "/ "
              << hours_predict << ":" << mins_predict << ":" << secs_predict << "]                \t\r";
    // std::cerr << "\r" << std::fixed << std::setprecision(3) << progress * 100 << "%"
    //           << " [" << hours << ":" << mins << ":" << secs << "/ "
    //           << hours_predict << ":" << mins_predict << ":" << secs_predict << "]                \t\r";

    // std::cerr.flush();
}

NAMESPACE_END(raytracer)