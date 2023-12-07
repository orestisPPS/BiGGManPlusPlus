//
// Created by hal9000 on 11/21/23.
//

#ifndef UNTITLED_TIMER_H
#define UNTITLED_TIMER_H

#include <chrono>

class Timer {

public:
    explicit Timer();
    
    void start();

    void stop();
    
    std::chrono::duration<double> duration() const;
    
private:
    bool _running;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
    std::chrono::time_point<std::chrono::high_resolution_clock> _end;
};

#endif //UNTITLED_TIMER_H
