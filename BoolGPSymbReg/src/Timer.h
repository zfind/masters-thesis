#pragma once

#include <chrono>

class Timer {
public:
    Timer()
            :diff(0L)
    {
    }

    ~Timer() = default;

    void start()
    {
        begin = std::chrono::steady_clock::now();
    }

    void pause()
    {
        end = std::chrono::steady_clock::now();
        diff += std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    }

    long get()
    {
        return diff;
    }

    void reset()
    {
        diff = 0L;
    }

private:
    long diff;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

};
