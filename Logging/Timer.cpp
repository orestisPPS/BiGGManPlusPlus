//
// Created by hal9000 on 11/21/23.
//

#include "Timer.h"

Timer::Timer(std::string loggedName) {
    _name = std::move(loggedName);
    _running = false;
}

void Timer::start() {
    if (_running) {
        std::cout << "Timer " << _name << " is already running." << std::endl;
        return;
    }
    _start = std::chrono::high_resolution_clock::now();
    _running = true;
}

void Timer::stop() {
    if (!_running) {
        std::cout << "Timer " << _name << " is not running." << std::endl;
        return;
    }
    _end = std::chrono::high_resolution_clock::now();
    _duration = _end - _start;
}

std::chrono::duration<double> Timer::duration() const {
    return _duration;
}