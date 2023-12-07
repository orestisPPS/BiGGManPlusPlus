//
// Created by hal9000 on 10/22/23.
//

#ifndef UNTITLED_TRANSIENTANALYSISRESULTS_H
#define UNTITLED_TRANSIENTANALYSISRESULTS_H
#include <map>
#include "../../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalVector/NumericalVector.h"

namespace NumericalAnalysis {

    class TransientAnalysisResults {
    public:
        TransientAnalysisResults(double initialTime, unsigned int totalSteps, double stepSize, unsigned numberOfDof, bool storeDerivative1 = false, bool storeDerivative2 = false);
        
        void addSolution(unsigned stepIndex, shared_ptr<NumericalVector<double>> solution,
                         const shared_ptr<NumericalVector<double>>& derivative1 = nullptr,
                         const shared_ptr<NumericalVector<double>>& derivative2 = nullptr);
        
        unique_ptr<NumericalVector<double>> getSolutionAtDof(unsigned derivativeOrder, unsigned dofID) const;
    private:
        struct TimeStepData{
            double timeValue;
            vector<shared_ptr<NumericalVector<double>>> solution;
        };
        
        unique_ptr<vector<TimeStepData>> _solution;
        bool _storeDerivative1;
        bool _storeDerivative2;
        unsigned _totalSteps;

    };

} // NumericalAnalysis

#endif //UNTITLED_TRANSIENTANALYSISRESULTS_H