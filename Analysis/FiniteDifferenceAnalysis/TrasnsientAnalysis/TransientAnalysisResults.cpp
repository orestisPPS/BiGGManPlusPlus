//
// Created by hal9000 on 10/22/23.
//

#include "TransientAnalysisResults.h"

namespace NumericalAnalysis {
    TransientAnalysisResults::TransientAnalysisResults(double initialTime, unsigned int totalSteps, double stepSize, unsigned numberOfDof, bool storeDerivative1, bool storeDerivative2) {
        _storeDerivative1 = storeDerivative1;
        _storeDerivative2 = storeDerivative2;
        _solution = make_unique<map<unsigned , map<unsigned, shared_ptr<NumericalVector<double>>>>>();
        double currentTime = initialTime;
        
        
        
        
        for (unsigned int i = 0; i < totalSteps; ++i) {
            _solution->insert(make_pair(i, map<unsigned, shared_ptr<NumericalVector<double>>>()));
            _solution->at(i).insert(make_pair(0, make_shared<NumericalVector<double>>(numberOfDof)));
            if (storeDerivative1){
                _solution->at(i).insert(make_pair(1, make_shared<NumericalVector<double>>(numberOfDof)));
            }
            if (storeDerivative2){
                _solution->at(i).insert(make_pair(2, make_shared<NumericalVector<double>>(numberOfDof)));
            }
            currentTime += stepSize;
        }
    }

    void TransientAnalysisResults::addSolution(unsigned time, shared_ptr<NumericalVector<double>> solution,
                                               shared_ptr<NumericalVector<double>> derivative1,
                                               shared_ptr<NumericalVector<double>> derivative2) {
        _solution->at(time).at(0)->deepCopy(solution);

        if (_storeDerivative1 && derivative1 != nullptr){
            _solution->at(time).at(1)->deepCopy(derivative1);
        }
        if (_storeDerivative1 && derivative1 == nullptr){
            throw runtime_error("TransientAnalysisResults::addSolution - Expected non-null derivative1 as _storeDerivative1 is true, but received nullptr.");
        }
        if (_storeDerivative2 && derivative2 != nullptr){
            _solution->at(time).at(2)->deepCopy(derivative2);
        }
        if (_storeDerivative2 && derivative2 == nullptr){
            throw runtime_error("TransientAnalysisResults::addSolution - Expected non-null derivative2 as _storeDerivative2 is true, but received nullptr.");
        }
    }



} // NumericalAnalysis