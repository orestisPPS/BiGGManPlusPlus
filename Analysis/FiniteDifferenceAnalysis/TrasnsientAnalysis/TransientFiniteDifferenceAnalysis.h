//
// Created by hal9000 on 10/6/23.
//

#ifndef UNTITLED_TRANSIENTFINITEDIFFERENCEANALYSIS_H
#define UNTITLED_TRANSIENTFINITEDIFFERENCEANALYSIS_H
#include "TransientLinearSystemBuilder.h"
#include "../SteadyStateFiniteDifferenceAnalysis.h"
#include "../../../MathematicalEntities/MathematicalProblem/TransientMathematicalProblem.h"
#include "../../../LinearAlgebra/NumericalIntegrators/NumericalIntegrator.h"

namespace NumericalAnalysis {

    class TransientFiniteDifferenceAnalysis : public FiniteDifferenceAnalysis{
    public:
        TransientFiniteDifferenceAnalysis(double initialTime, double stepSize, unsigned totalSteps,
                                          shared_ptr<TransientMathematicalProblem> mathematicalProblem,
                                          shared_ptr<Mesh> mesh,
                                          shared_ptr<Solver> solver, shared_ptr<NumericalIntegrator> integrationMethod,
                                          shared_ptr<FDSchemeSpecs> schemeSpecs, CoordinateType coordinateSystem = Natural);
        
        
        void solve() override;
        
    private:
        shared_ptr<TransientMathematicalProblem> _transientMathematicalProblem;
        
        shared_ptr<NumericalIntegrator> _integrationMethod;
        
        double _initialTime;
        double _finalTime;
        double _stepSize;
        unsigned _totalSteps;
        
        shared_ptr<NumericalMatrix<double>> _M;
        shared_ptr<NumericalMatrix<double>> _C;
        shared_ptr<NumericalMatrix<double>> _K;
        shared_ptr<NumericalVector<double>> _RHS;
        
        
        void _assembleConstitutiveMatrices();
    };

} // NumericalAnalysis  


#endif //UNTITLED_TRANSIENTFINITEDIFFERENCEANALYSIS_H
