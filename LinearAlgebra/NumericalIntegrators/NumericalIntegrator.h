//
// Created by hal9000 on 10/8/23.
//

#ifndef UNTITLED_NUMERICALINTEGRATOR_H
#define UNTITLED_NUMERICALINTEGRATOR_H
#include "../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../MathematicalEntities/MathematicalProblem/TransientMathematicalProblem.h"
#include "../Solvers/Solver.h"
#include "../../ThreadingOperations/ThreadingOperations.h"
#include "../../Discretization/Time/TimeEntity.h"
#include "../../Analysis/FiniteDifferenceAnalysis/TrasnsientAnalysis/TransientAnalysisResults.h"
using namespace NumericalAnalysis;
namespace LinearAlgebra {

    class NumericalIntegrator {
    public:
        NumericalIntegrator();
        
        void setTimeParameters(double initialTime, double finalTime, unsigned int totalSteps);
        
        void setSolver(shared_ptr<Solver> solver);
        
        const shared_ptr<Solver> & getSolver() const;
        
        void setMatricesAndVector(shared_ptr<NumericalMatrix<double>> & M, shared_ptr<NumericalMatrix<double>> & C,
                                 shared_ptr<NumericalMatrix<double>> & K, shared_ptr<NumericalVector<double>> & RHS);

        void setMatricesAndVector(shared_ptr<NumericalMatrix<double>> & C, shared_ptr<NumericalMatrix<double>> & K,
                                 shared_ptr<NumericalVector<double>> & RHS);

        shared_ptr<LinearSystem> assembleEffectiveLinearSystem();
        
        shared_ptr<TransientAnalysisResults> results;

        virtual void assembleEffectiveMatrix();

        virtual void assembleEffectiveRHS();
        
        virtual void solveCurrentTimeStep(unsigned stepIndex, double currentTime, double currentTimeStep);
        
        virtual void solveCurrentTimeStepWithMatrixRebuild();

        shared_ptr<Solver> _solver;


        
    protected:
        double _initialTime;
        double _finalTime;
        double _timeStep;
        double _currentTime;
        unsigned int _numberOfTimeSteps;
        unsigned int _currentStep;

        shared_ptr<NumericalMatrix<double>> _M;
        shared_ptr<NumericalMatrix<double>> _C;
        shared_ptr<NumericalMatrix<double>> _K;
        shared_ptr<NumericalVector<double>> _RHS;
        shared_ptr<NumericalMatrix<double>> _K_hat;
        shared_ptr<NumericalVector<double>> _RHS_hat;
        
        shared_ptr<NumericalVector<double>> _U_old;
        shared_ptr<NumericalVector<double>> _U_dot_old;
        shared_ptr<NumericalVector<double>> _U_dot_dot_old;

        shared_ptr<NumericalVector<double>> _U_new;
        shared_ptr<NumericalVector<double>> _U_dot_new;
        shared_ptr<NumericalVector<double>> _U_dot_dot_new;
        
        shared_ptr<NumericalVector<double>> _sum;
        shared_ptr<NumericalVector<double>> _matrixVectorProduct1;
        shared_ptr<NumericalVector<double>> _matrixVectorProduct2;
         
        shared_ptr<LinearSystem> _currentLinearSystem;

        
        bool _timeParametersSet = false;
        bool _dataSet = false;
        bool _solverSet = false;
        bool _effectiveLinearSystemAssembled = false;


        
        virtual void _calculateFirstOrderDerivative();
        virtual void _calculateSecondOrderDerivative();
        virtual void _calculateHigherOrderDerivatives();
        virtual void _calculateIntegrationCoefficients();
        
        void _initializeMethodVectors();
        
        void _checkSolver() const;
        void _checkTimeParameters() const;
        void _checkData() const;
    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALINTEGRATOR_H
