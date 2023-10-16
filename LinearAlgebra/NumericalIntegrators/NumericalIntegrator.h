//
// Created by hal9000 on 10/8/23.
//

#ifndef UNTITLED_NUMERICALINTEGRATOR_H
#define UNTITLED_NUMERICALINTEGRATOR_H
#include "../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../MathematicalEntities/MathematicalProblem/TransientMathematicalProblem.h"
#include "../Solvers/Solver.h"
namespace LinearAlgebra {

    class NumericalIntegrator {
    public:
        NumericalIntegrator(double initialTime, double finalTime, double timeStep,
                            const shared_ptr<TransientMathematicalProblem>& mathematicalProblem,
                            const shared_ptr<Solver>& solver);
    private:
        double _initialTime;
        double _finalTime;
        double _timeStep;

        shared_ptr<NumericalMatrix<double>> _M;
        shared_ptr<NumericalMatrix<double>> _C;
        shared_ptr<NumericalMatrix<double>> _K;
        shared_ptr<NumericalMatrix<double>> _K_hat;
        shared_ptr<TransientMathematicalProblem> _mathematicalProblem;
        shared_ptr<Solver> _solver;

        void _assembleEffectiveMatrix();
        void _assembleEffectiveRHS();
    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALINTEGRATOR_H
