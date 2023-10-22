
//
// Created by hal9000 on 10/6/23.
//

#include "TransientFiniteDifferenceAnalysis.h"

namespace NumericalAnalysis {
    TransientFiniteDifferenceAnalysis::TransientFiniteDifferenceAnalysis(double initialTime, double stepSize, unsigned totalSteps,
            shared_ptr<TransientMathematicalProblem> mathematicalProblem,
            shared_ptr<Mesh> mesh,
            shared_ptr<Solver> solver,
            shared_ptr<NumericalIntegrator> integrationMethod,
            shared_ptr<FDSchemeSpecs> schemeSpecs,
            CoordinateType coordinateSystem) :
            FiniteDifferenceAnalysis(mathematicalProblem, std::move(mesh), std::move(solver), std::move(schemeSpecs), coordinateSystem) {

        _initialTime = initialTime;
        _finalTime = initialTime + totalSteps * stepSize;
        _stepSize = stepSize;
        _totalSteps = totalSteps;
        _transientMathematicalProblem = std::move(mathematicalProblem);
        _integrationMethod = std::move(integrationMethod);
        _integrationMethod->setSolver(this->solver);
    }
    
    void TransientFiniteDifferenceAnalysis::solve(){
        _assembleConstitutiveMatrices();
        _integrationMethod->setMatricesAndVector(_M, _C, _K, _RHS);
        _integrationMethod->setTimeParameters(_initialTime, _finalTime, _totalSteps);
        for (unsigned int i = 0; i < _totalSteps; ++i) {
            _integrationMethod->solveCurrentTimeStep(i, _initialTime + i * _stepSize, _stepSize);
            cout << "Time step " << i << " solved" << endl;
        }
        
    }

    void TransientFiniteDifferenceAnalysis::_assembleConstitutiveMatrices() {
        auto linearSystemInitializer = TransientLinearSystemBuilder(degreesOfFreedom, mesh, _transientMathematicalProblem,
                                                                    schemeSpecs, coordinateSystem);
        linearSystemInitializer.assembleMatrices();
        _M = std::move(linearSystemInitializer.M);
        _C = std::move(linearSystemInitializer.C);
        _K = std::move(linearSystemInitializer.K);
        _RHS = std::move(linearSystemInitializer.RHS);
    }

}
