#pragma once

#include <ctime>
#include <cstdio>
#include <iostream>
#include <string>

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
	void PrintTimePassed(const std::string &name)
	{
		time_t now;
		time(&now);
		auto diff = difftime(now, start_);
		auto pass_hours = static_cast<int>(diff / 3600);
		auto pass_mins = static_cast<int>(diff / 60) - pass_hours * 60;
		auto pass_secs = static_cast<int>(diff) - pass_hours * 3600 - pass_mins * 60;
		printf("[info] finished \"%s\" at %s\t %i h %i m %i s have passed so far.\n", name.c_str(), ctime(&now), pass_hours, pass_mins, pass_secs);
	}
	//\brief 在命令行输出从计时器开始计时起流逝的时间
	void PrintTimePassed2(const std::string &name)
	{
		time_t now;
		time(&now);
		auto diff = difftime(now, start_);
		auto pass_hours = static_cast<int>(diff / 3600);
		auto pass_mins = static_cast<int>(diff / 60) - pass_hours * 60;
		auto pass_secs = static_cast<int>(diff) - pass_hours * 3600 - pass_mins * 60;
		printf("[info] it takes %i h %i m %i s for \"%s\".\n", pass_hours, pass_mins, pass_secs, name.c_str());
	}
};