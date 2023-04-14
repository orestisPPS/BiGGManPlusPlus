//
// Created by hal9000 on 4/14/23.
//

#include "VectorOperations.h"

namespace LinearAlgebra {
    
        double VectorOperations::dotProduct(vector<double> *vector1, vector<double> *vector2) {
            auto result = 0.0;
            if (vector1->size() != vector2->size())
                throw std::invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1->size(); i++)
                result += vector1->at(i) * vector2->at(i);
            return result;
        }
        
        double VectorOperations::dotProduct(vector<double> &vector1, vector<double> &vector2) {
            auto result = 0.0;
            if (vector1.size() != vector2.size())
                throw std::invalid_argument("Vectors must have the same size");
            for (auto i = 0; i < vector1.size(); i++)
                result += vector1.at(i) * vector2.at(i);
            return result;
        }
        
        vector<double> VectorOperations::crossProduct(vector<double>* vector1, vector<double>* vector2) {
            switch (vector1->size()) {
                case 2:
                    if (vector2->size() != 2)
                        throw std::invalid_argument("Vectors must have 2 dimensions");
                    return {vector1->at(0) * vector2->at(1) - vector1->at(1) * vector2->at(0)};
                case 3:
                    if (vector2->size() != 3)
                        throw std::invalid_argument("Vectors must have 3 dimensions");
                    return {vector1->at(1) * vector2->at(2) - vector1->at(2) * vector2->at(1),
                            vector1->at(2) * vector2->at(0) - vector1->at(0) * vector2->at(2),
                            vector1->at(0) * vector2->at(1) - vector1->at(1) * vector2->at(0)};
                default:
                    throw std::invalid_argument("Vectors must have 2 or 3 dimensions");
            }
        }
        
        vector<double> VectorOperations::crossProduct(vector<double> &vector1, vector<double> &vector2) {
            switch (vector1.size()) {
                case 2:
                    if (vector2.size() != 2)
                        throw std::invalid_argument("Vectors must have 2 dimensions");
                    return {vector1.at(0) * vector2.at(1) - vector1.at(1) * vector2.at(0)};
                case 3:
                    if (vector2.size() != 3)
                        throw std::invalid_argument("Vectors must have 3 dimensions");
                    return {vector1.at(1) * vector2.at(2) - vector1.at(2) * vector2.at(1),
                            vector1.at(2) * vector2.at(0) - vector1.at(0) * vector2.at(2),
                            vector1.at(0) * vector2.at(1) - vector1.at(1) * vector2.at(0)};
                default:
                    throw std::invalid_argument("Vectors must have 2 or 3 dimensions");
            }
        }
} // LinearAlgebra