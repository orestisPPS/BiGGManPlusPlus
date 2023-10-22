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
        
        void addSolution(unsigned time, shared_ptr<NumericalVector<double>> solution, shared_ptr<NumericalVector<double>> derivative1 = nullptr,
                                                                                    shared_ptr<NumericalVector<double>> derivative2 = nullptr);
    private:
        unique_ptr<NumericalVector<double>> _timeStamps;
        unique_ptr<map<unsigned, map<unsigned, shared_ptr<NumericalVector<double>>>>> _solution;
        bool _storeDerivative1;
        bool _storeDerivative2;
    };

} // NumericalAnalysis

#endif //UNTITLED_TRANSIENTANALYSISRESULTS_H
