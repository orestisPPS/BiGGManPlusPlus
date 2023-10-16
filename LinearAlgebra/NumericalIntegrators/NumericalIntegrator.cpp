//
// Created by hal9000 on 10/8/23.
//

#include "NumericalIntegrator.h"

namespace LinearAlgebra {
    
        NumericalIntegrator::NumericalIntegrator(double initialTime, double finalTime, double timeStep,
                                                 const shared_ptr<TransientMathematicalProblem>& mathematicalProblem,
                                                 const shared_ptr<Solver>& solver) :
                                                _initialTime(initialTime),
                                                _finalTime(finalTime),
                                                _timeStep(timeStep),
                                                _mathematicalProblem(std::move(mathematicalProblem)),
                                                _solver(std::move(solver)) {

        }
    
        void NumericalIntegrator::_assembleEffectiveMatrix() {

        }
    
        void NumericalIntegrator::_assembleEffectiveRHS() {
        }
    
} // LinearAlgebra