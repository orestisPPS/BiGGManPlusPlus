//
// Created by hal9000 on 4/14/23.
//

#ifndef UNTITLED_VECTOROPERATIONS_H
#define UNTITLED_VECTOROPERATIONS_H

#include "Array.h"

namespace LinearAlgebra {

    class VectorOperations {
    public:
        
        static double dotProduct(vector<double> *vector1, vector<double> *vector2);

        static double dotProduct(vector<double> &vector1, vector<double> &vector2);
        
        static vector<double> crossProduct(vector<double>* vector1, vector<double>* vector2);
        
        static vector<double> crossProduct(vector<double> &vector1, vector<double> &vector2);

    };

} // LinearAlgebra

#endif //UNTITLED_VECTOROPERATIONS_H
