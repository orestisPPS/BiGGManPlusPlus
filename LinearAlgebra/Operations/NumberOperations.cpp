//
// Created by hal9000 on 5/1/23.
//

#include "NumberOperations.h"

namespace LinearAlgebra {
    
        int NumberOperations::factorial(int n) {
            if (n == 0)
                return 1;
            else{
                auto result = 1;
                for (auto i = 1; i <= n; i++) {
                    result *= i;
                }
                return result;
            }
        }
        
        double NumberOperations::factorial(double n) {
            if (n == 0)
                return 1;
            else{
                auto result = 1;
                for (auto i = 1; i <= n; i++) {
                    result *= i;
                }
                return result;
            }
        }
    
        int NumberOperations::binomialCoefficient(int n, int k) {
            return factorial(n) / (factorial(k) * factorial(n - k));
        }
    
        double NumberOperations::binomialCoefficient(double n, double k) {
            return factorial(n) / (factorial(k) * factorial(n - k));
        }
} // LinearAlgebra