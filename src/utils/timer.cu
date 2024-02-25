#include "csrt/utils/timer.cuh"

#include <cstdio>

Timer::Timer() { time_begin_ = std::chrono::steady_clock::now(); }

void Timer::Reset() { time_begin_ = std::chrono::steady_clock::now(); }

void Timer::PrintTimePassed(const std::string &work_name)
{
    time_current_ = std::chrono::steady_clock::now();
    double diff =
        std::chrono::duration<double>(time_current_ - time_begin_).count();
    unsigned int hours = static_cast<unsigned int>(diff / 3600),
                 mins = static_cast<unsigned int>(diff / 60) - hours * 60,
                 secs =
                     static_cast<unsigned int>(diff) - hours * 3600 - mins * 60,
                 ms = static_cast<unsigned int>(
                     (diff - static_cast<unsigned int>(diff)) * 1000);
    fprintf(stderr,
            "[info] \"%s\" finished. \n\tIt takes %u hr %u min %u sec %u ms.\n",
            work_name.c_str(), hours, mins, secs, ms);
}

void Timer::PrintProgress(double progress)
{
    time_current_ = std::chrono::steady_clock::now();
    double diff =
        std::chrono::duration<double>(time_current_ - time_begin_).count();
    int hours = static_cast<int>(diff / 3600),
        mins = static_cast<int>(diff / 60) - hours * 60,
        secs = static_cast<int>(diff) - hours * 3600 - mins * 60;

    double diff_predict = (progress > 1e-4) ? diff / progress : 0;
    int hours_predict = static_cast<int>(diff_predict / 3600),
        mins_predict = static_cast<int>(diff_predict / 60) - hours_predict * 60,
        secs_predict = static_cast<int>(diff_predict) - hours_predict * 3600 -
                       mins_predict * 60;

    fprintf(stderr, "\r[info] %3.5f %%  [%d:%02d:%02d / %d:%02d:%02d] \t\r",
            progress * 100, hours, mins, secs, hours_predict, mins_predict,
            secs_predict);
}
