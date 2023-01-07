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
        static double Distance(vector<double> &pointOne, vector<double> &pointTwo) {
            double distance = 0;
            if (pointOne.size() != pointTwo.size()) {
                throw "Point dimensions do not match";
            }
            for (int i = 0; i < pointOne.size(); i++) {
                distance += pow(pointOne[i] - pointTwo[i], 2);
            }
            return sqrt(distance);
        }
        
        static double RadiansToDegrees(double radians) {
            return radians * 180 / M_PI;
        }
        
        static double DegreesToRadians(double degrees) {
            return degrees * M_PI / 180;
        }
    }


    };// Utility

#endif //UNTITLED_CALCULATORS_H
