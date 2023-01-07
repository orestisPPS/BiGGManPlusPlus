//
// Created by hal9000 on 1/7/23.
//

#include "Transformations.h"

namespace LinearAlgebra {
    Transformations::Transformations() = default;
    
    Array<double> Transformations::translateDirection1(vector<double> &vector, double amount) {
        Array<double> result(vector.size());
        for (int i = 0; i < vector.size(); i++) {
            result[i] = vector[i] + amount;
        }
        return result;
    }
    
} // LinearAlgebra