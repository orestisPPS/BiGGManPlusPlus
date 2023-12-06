//
// Created by hal9000 on 11/27/23.
//

#ifndef UNTITLED_CHRONODURATIONCONVERTERS_H
#define UNTITLED_CHRONODURATIONCONVERTERS_H


#include <chrono>

class ChronoDurationConverters {
public:
    static double convertToSeconds(const std::chrono::duration<double> &duration);
    
    static double convertToMilliseconds(const std::chrono::duration<double> &duration);
    
    static double convertToMicroseconds(const std::chrono::duration<double> &duration);
    
    static double convertToNanoseconds(const std::chrono::duration<double> &duration);
    
    static double convertToMinutes(const std::chrono::duration<double> &duration);
    
    static double convertToHours(const std::chrono::duration<double> &duration);
    
    static double convertToDays(const std::chrono::duration<double> &duration);
};


#endif //UNTITLED_CHRONODURATIONCONVERTERS_H
