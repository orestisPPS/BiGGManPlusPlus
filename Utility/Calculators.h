//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_CALCULATORS_H
#define UNTITLED_CALCULATORS_H

#include <vector>
#include <cmath>
using namespace std;

namespace Utility {
    namespace Calculators {

        
        static double radiansToDegrees(double radians) {
            return radians * 180 / M_PI;
        }
        
        static double degreesToRadians(double degrees) {
            return degrees * M_PI / 180;
        }

    }


    };// Utility

#endif //UNTITLED_CALCULATORS_H
