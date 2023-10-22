//
// Created by hal9000 on 10/22/23.
//

#include "TransientAnalysisResults.h"

namespace NumericalAnalysis {
    TransientAnalysisResults::TransientAnalysisResults(double initialTime, unsigned int totalSteps, double stepSize, unsigned numberOfDof, bool storeDerivative1, bool storeDerivative2) {
        _storeDerivative1 = storeDerivative1;
        _storeDerivative2 = storeDerivative2;
        _solution = make_unique<vector<TimeStepData>>(totalSteps);
        double currentTime = initialTime;
        
        for (unsigned int i = 0; i < totalSteps; ++i) {

            (*_solution)[i].timeValue = currentTime;
            (*_solution)[i].solution = vector<shared_ptr<NumericalVector<double>>>(3);
            for (auto & vector : (*_solution)[i].solution){
                vector = make_shared<NumericalVector<double>>(numberOfDof);
            }
            currentTime += stepSize;
        }
    }

    void TransientAnalysisResults::addSolution(unsigned stepIndex, shared_ptr<NumericalVector<double>> solution,
                                               const shared_ptr<NumericalVector<double>>& derivative1,
                                               const shared_ptr<NumericalVector<double>>& derivative2) {
        (*_solution)[stepIndex].solution[0]->deepCopy(solution);
        if (_storeDerivative1 and not _storeDerivative2){
            (*_solution)[stepIndex].solution[1]->deepCopy(derivative1);
            (*_solution)[stepIndex].solution[2] = nullptr;
        }
        if (_storeDerivative1 and _storeDerivative2){
            (*_solution)[stepIndex].solution[1]->deepCopy(derivative1);
            (*_solution)[stepIndex].solution[2]->deepCopy(derivative2);
        }
        if (not _storeDerivative1 and _storeDerivative2){
            (*_solution)[stepIndex].solution[1] = nullptr;
            (*_solution)[stepIndex].solution[2]->deepCopy(derivative2);
        }
        if (not _storeDerivative1 and not _storeDerivative2){
            (*_solution)[stepIndex].solution[1] = nullptr;
            (*_solution)[stepIndex].solution[2] = nullptr;
        }
    }



} // NumericalAnalysis