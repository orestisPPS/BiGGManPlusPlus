//
// Created by hal9000 on 11/21/23.
//

#ifndef UNTITLED_TIMER_H
#define UNTITLED_TIMER_H

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <map>

class Timer {

public:
    explicit Timer(std::string loggedName);

    std::string _name;

    void start();

    void stop();
    
    std::chrono::duration<double> duration() const;
    

    
private:
    bool _running;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
    std::chrono::time_point<std::chrono::high_resolution_clock> _end;
    std::chrono::duration<double> _duration;
};

#endif //UNTITLED_TIMER_H
