//
// Created by hal9000 on 11/27/23.
//

#include "ChronoDurationConverters.h"

double ChronoDurationConverters::convertToSeconds(const std::chrono::duration<double> &duration) {
    return duration.count();
}

double ChronoDurationConverters::convertToMilliseconds(const std::chrono::duration<double> &duration) {
    return duration.count() * 1E3;
}

double ChronoDurationConverters::convertToMicroseconds(const std::chrono::duration<double> &duration) {
    return duration.count() * 1E6;
}

double ChronoDurationConverters::convertToNanoseconds(const std::chrono::duration<double> &duration) {
    return duration.count() * 1E9;
}

double ChronoDurationConverters::convertToMinutes(const std::chrono::duration<double> &duration) {
    return duration.count() / 60.0;
}

double ChronoDurationConverters::convertToHours(const std::chrono::duration<double> &duration) {
    return duration.count() / 3600.0;
}

double ChronoDurationConverters::convertToDays(const std::chrono::duration<double> &duration) {
    return duration.count() / 86400.0;
}





