//
// Created by hal9000 on 1/7/23.
//

#include "Calculators.h"

namespace Utility {
    
        Calculators::Calculators() = default;
    
        double Calculators::degreesToRadians(double degrees) {
            return degrees * (M_PI / 180);
        }
    
        double Calculators::radiansToDegrees(double radians) {
            return radians * (180 / M_PI);
        }
    
        double Calculators::calculateAngleBetweenTwoVectors(vector<double> vectorOne, vector<double> vectorTwo) {
            double dotProduct = 0;
            double magnitudeOne = 0;
            double magnitudeTwo = 0;
            for (int i = 0; i < vectorOne.size(); i++) {
                dotProduct += vectorOne[i] * vectorTwo[i];
                magnitudeOne += vectorOne[i] * vectorOne[i];
                magnitudeTwo += vectorTwo[i] * vectorTwo[i];
            }
            return acos(dotProduct / (sqrt(magnitudeOne) * sqrt(magnitudeTwo)));
        }
    
    } // Utility
} // Utility