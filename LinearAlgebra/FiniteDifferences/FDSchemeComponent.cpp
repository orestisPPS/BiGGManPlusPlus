//
// Created by hal9000 on 4/15/23.
//

#include "FDSchemeComponent.h"

namespace LinearAlgebra {
    
        FDSchemeComponent::FDSchemeComponent(FiniteDifferenceSchemeType schemeType, unsigned errorOrder,
                                            map<int, double>* functionValues, map<int, double>* weights, double stepSize) {
            checkInput(functionValues, weights);
            this->schemeType = schemeType;
            this->errorOrder = errorOrder;
            this->functionValues = functionValues;
            this->weights = weights;
            this->stepFractions = new map<int, double>();
            for (auto weight : *weights) {
                this->stepFractions->insert(make_pair(weight.first, 1.0 / stepSize));
            }
            this->values = calculateValues();
        }
    
        FDSchemeComponent::FDSchemeComponent(FiniteDifferenceSchemeType schemeType, unsigned errorOrder,
                                            map<int, double>* functionValues, map<int, double>* weights,
                                            map<int, double>* stepFractions) {
            checkInput(functionValues, weights, stepFractions);
            this->schemeType = schemeType;
            this->errorOrder = errorOrder;
            this->functionValues = functionValues;
            this->weights = weights;
            this->stepFractions = stepFractions;
            this->values = calculateValues();
        }
        
        FDSchemeComponent::~FDSchemeComponent() {
            delete functionValues;
            delete weights;
            delete stepFractions;
            delete values;
        }
        
        void FDSchemeComponent::checkInput(map<int, double>* inputFunctionValues, map<int, double>* inputWeights) {
            if (inputWeights->size() != inputFunctionValues->size()) {
                throw invalid_argument("The number of weights and function values must be the same!");
            }
            for (auto weight : *inputWeights) {
                auto& index = weight.first;
                if (inputFunctionValues->find(index) == inputFunctionValues->end()) {
                    throw invalid_argument("The function values must contain the index of the weight!");
                }
            }
        }
        
        void FDSchemeComponent::checkInput(map<int, double>* inputFunctionValues, map<int, double>* inputWeights,
                                           map<int, double>* inputStepFractions) {
            if (inputWeights->size() != inputFunctionValues->size() && inputWeights->size() != inputStepFractions->size()) {
                throw invalid_argument("The number of weights, function values and step fractions must be the same!");
            }
            for (auto weight : *inputWeights) {
                auto& index = weight.first;
                if (inputFunctionValues->find(index) == inputFunctionValues->end()) {
                    throw invalid_argument("The function values must contain the index of the weight!");
                }
            }
        }

        map<int, double> *FDSchemeComponent::calculateValues() const {
            auto schemeValues = new map<int, double>();
            for (auto &weight : *weights) {
                auto &index = weight.first;
                //Value = weight * functionValue * stepFraction
                values->insert(pair<int, double>(
                        weight.first, weight.second * (*functionValues)[index] * (*stepFractions)[index]));
            }
            return schemeValues;
        }


} // LinearAlgebra