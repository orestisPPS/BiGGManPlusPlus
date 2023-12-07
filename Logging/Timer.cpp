//
// Created by hal9000 on 11/21/23.
//

#include <iostream>
#include "Timer.h"

Timer::Timer() : _running(false) { }

void Timer::start() {
    if (_running) {
        throw std::runtime_error("Timer::start: Timer is already running.");
        return;
    }
    _start = std::chrono::high_resolution_clock::now();
    _running = true;
}

void Timer::stop() {
    if (!_running) {
        std::cout << "Timer::stop: Timer is not running." << std::endl;
        return;
    }
    _end = std::chrono::high_resolution_clock::now();
    _running = false;
}

std::chrono::duration<double> Timer::duration() const {
    return _end - _start;
}