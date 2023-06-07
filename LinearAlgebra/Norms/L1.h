//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_L1_H
#define UNTITLED_L1_H

#include "VectorNorm.h"

namespace LinearAlgebra {

    class L1 : public VectorNorm {
    public:
        L1(vector<double>* vector, VectorNormType normType) : VectorNorm(vector, normType) {

        }
        
        double calculateNorm() override {
            double norm = 0;
            for (int i = 0; i < vector->size(); i++) {
                norm += abs(vector->at(i));
            }
            return norm;
        }

    };

} // LinearAlgebra

#endif //UNTITLED_L1_H
