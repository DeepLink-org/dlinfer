#pragma once

#include <chrono>
#include <iostream>
#include <thread>

namespace dicp {

class Timer {
public:
    Timer() : startTimePoint(), endTimePoint(), isRunning(false) {}

    void start() {
        startTimePoint = std::chrono::high_resolution_clock::now();
        isRunning = true;
    }

    void stop() {
        if (isRunning) {
            endTimePoint = std::chrono::high_resolution_clock::now();
            isRunning = false;
        }
    }

    double ElapsedMicroSecond() const {
        if (isRunning) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTimePoint).count();
        } else {
            return std::chrono::duration_cast<std::chrono::microseconds>(endTimePoint - startTimePoint).count();
        }
    }

    double ElapsedSecond() const { return ElapsedMicroSecond() / 1e6; }

    void reset() {
        startTimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>();
        endTimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>();
        isRunning = false;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;
    std::chrono::time_point<std::chrono::high_resolution_clock> endTimePoint;
    bool isRunning;
};

}  // namespace dicp
