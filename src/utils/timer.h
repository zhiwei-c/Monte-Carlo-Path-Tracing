#pragma once

#include <ctime>
#include <cstdio>
#include <iostream>

NAMESPACE_BEGIN(simple_renderer)

//计时器类
class Timer
{
private:
	time_t start_; //开始计时的时刻

public:
	Timer() { time(&start_); }

	//\breif 重新开始计时
	void Reset() { time(&start_); }

	//\brief 在命令行输出从计时器开始计时起流逝的时间
	void PrintTimePassed()
	{
		time_t now;
		time(&now);
		auto diff = difftime(now, start_);
		auto hours = static_cast<int>(diff / 3600);
		auto mins = static_cast<int>(diff / 60) - hours * 60;
		auto secs = static_cast<int>(diff) - hours * 3600 - mins * 60;
		printf("Finished. It takes %i h, %i m, %i s.\n", hours, mins, secs);
	}

	//\brief 在命令行输出工作进度，已耗时和预估总耗时
	//
	//\param progress 工作进度，范围[0,1]
	void PrintProgress(Float progress)
	{
		time_t now;
		time(&now);

		auto diff = difftime(now, start_);
		auto hours = static_cast<int>(diff / 3600);
		auto mins = static_cast<int>(diff / 60) - hours * 60;
		auto secs = static_cast<int>(diff) - hours * 3600 - mins * 60;

		auto diff_pred = (progress > kEpsilon) ? diff / progress : 0;
		auto hours_pred = static_cast<int>(diff_pred / 3600);
		auto mins_pred = static_cast<int>(diff_pred / 60) - hours_pred * 60;
		auto secs_pred = static_cast<int>(diff_pred) - hours_pred * 3600 - mins_pred * 60;

		printf("\r%.3f%% [%i:%i:%i / %i:%i:%i]                \t\r", progress * 100,
			   hours, mins, secs,
			   hours_pred, mins_pred, secs_pred);

		std::cout.flush();
	}
};

NAMESPACE_END(simple_renderer)