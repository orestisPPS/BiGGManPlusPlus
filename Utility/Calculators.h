//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_CALCULATORS_H
#define UNTITLED_CALCULATORS_H

#include <vector>
#include <cmath>
using namespace std;

namespace Utility {

    class Calculators {
    public:
        Calculators();
        double degreesToRadians(double degrees);
        double radiansToDegrees(double radians);
        double calculateAngleBetweenTwoVectors(vector<double> vectorOne, vector<double> vectorTwo);
    };

    };

} // Utility

#endif //UNTITLED_CALCULATORS_H
