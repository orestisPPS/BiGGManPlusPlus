//
// Created by hal9000 on 4/18/23.
//

#include "VectorNorm.h"

namespace LinearAlgebra {
    
        VectorNorm::VectorNorm(vector<double>* vector, VectorNormType normType): _normType(normType)   {
            switch (_normType) {
                case L1:
                    _value = _calculateL1Norm(vector);
                    break;
                case L2:
                    _value = _calculateL2Norm(vector);
                    break;
                case LInf:
                    _value = _calculateLInfNorm(vector);
                    break;
                default:
                    throw invalid_argument("Invalid norm type.");
            }
            
            //auto lol = _normTypeToFunction[L1](vector);
        }

        VectorNorm::VectorNorm(vector<double>* vector, VectorNormType normType, unsigned order): _normType(normType)   {
            if (_normType == Lp) {
                _value = _calculateLpNorm(vector, order);
            } else {
                throw invalid_argument("Only Lp norm can be calculated with an order.");
            }
        }
        
        VectorNormType & VectorNorm::type() {
            return _normType;
        }
    
        double VectorNorm::value() const {
            return _value;
        }
    
        double VectorNorm::_calculateL1Norm(vector<double>* vector) {
            double norm = 0;
            for (auto& component : *vector) {
                norm += abs(component);
            }
            return norm;
        }
    
        double VectorNorm::_calculateL2Norm(vector<double>* vector) {
            double norm = 0;
            for (auto& component : *vector) {
                norm += pow(component, 2);
            }
            return sqrt(norm);
        }
    
        double VectorNorm::_calculateLInfNorm(vector<double>* vector) {
            double norm = 0;
            for (auto& component : *vector) {
                norm = max(norm, abs(component));
            }
            return norm;
        }
    
        double VectorNorm::_calculateLpNorm(vector<double>* vector, double order) {
            double norm = 0;
            for (auto& component : *vector) {
                norm += pow(abs(component), order);
            }
            return pow(norm, 1 / order);
        }
    

} // LinearAlgebra