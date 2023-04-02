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
        static double distance(vector<double> &pointOne, vector<double> &pointTwo) {
            double distance = 0;
            if (pointOne.size() != pointTwo.size()) {
                throw "Point dimensions do not match";
            }
            for (int i = 0; i < pointOne.size(); i++) {
                distance += pow(pointOne[i] - pointTwo[i], 2);
            }
            return sqrt(distance);
        }
        
        static double radiansToDegrees(double radians) {
            return radians * 180 / M_PI;
        }
        
        static double degreesToRadians(double degrees) {
            return degrees * M_PI / 180;
        }
        
        // Calculates the magnitude of a vector
        static double magnitude(std::vector<double> vector) {
            double magnitude = 0;
            for (double i : vector) {
                magnitude += pow(i, 2);
            }
            return sqrt(magnitude);
        }
    }


    };// Utility

#endif //UNTITLED_CALCULATORS_H
