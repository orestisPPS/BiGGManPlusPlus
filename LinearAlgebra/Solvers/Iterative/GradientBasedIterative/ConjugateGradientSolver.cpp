//
// Created by hal9000 on 7/27/23.
//

#include "ConjugateGradientSolver.h"


namespace LinearAlgebra {
    ConjugateGradientSolver::ConjugateGradientSolver(double tolerance, unsigned maxIterations, VectorNormType normType,
                                                     unsigned userDefinedThreads, bool printOutput,
                                                     bool throwExceptionOnMaxFailure) :
            IterativeSolver(tolerance, maxIterations, normType, userDefinedThreads, printOutput,
                            throwExceptionOnMaxFailure) {
        _solverName = "Conjugate Gradient";
    }

    void ConjugateGradientSolver::_initializeVectors() {
        auto n = _linearSystem->matrix->numberOfRows();
        _xNew = make_shared<NumericalVector<double>>(n);
        _xOld = make_shared<NumericalVector<double>>(n);
        _residualOld = make_shared<NumericalVector<double>>(n);
        _residualNew = make_shared<NumericalVector<double>>(n);
        _directionVectorNew = make_shared<NumericalVector<double>>(n);
        _directionVectorOld = make_shared<NumericalVector<double>>(n);
        _difference = nullptr;
        _matrixVectorMultiplication = make_shared<NumericalVector<double>>(n);
        _vectorsInitialized = true;

    }
    

    void ConjugateGradientSolver::_cudaSolution() {
        IterativeSolver::_cudaSolution();
    }

    void ConjugateGradientSolver::_performMethodSolution() {
        if (printOutput)
            _printInitializationText();

        _alpha = 0.0;
        _beta = 0.0;
        _exitNorm = 1.0;
        //A * x_old
        _linearSystem->matrix->multiplyVector(_xOld, _matrixVectorMultiplication);
        //r_old = b - A * x_old
        _linearSystem->rhs->subtract(_matrixVectorMultiplication, _residualOld);
        double initialNorm = _residualOld->norm(L2);
        //d_old = r_old
        _directionVectorOld->deepCopy(_residualOld);
        
        while (_iteration < _maxIterations) {
            logs.startMultipleObservationsTimer("CG Iteration Time");
            
            _matrixVectorMultiplication->fill(0.0);
            _linearSystem->matrix->multiplyVector(_directionVectorOld, _matrixVectorMultiplication);
            double r_oldT_r_old = _residualOld->dotProduct(_residualOld, _userDefinedThreads);
            double direction_oldT_A_direction_old = _directionVectorOld->dotProduct(_matrixVectorMultiplication);
            _alpha = r_oldT_r_old / direction_oldT_A_direction_old;
            _xOld->add(_directionVectorOld, _xNew, 1.0, _alpha, _userDefinedThreads);
            _residualOld->subtract(_matrixVectorMultiplication, _residualNew, 1.0, _alpha, _userDefinedThreads);
            _exitNorm = _residualNew->norm(_normType) / initialNorm;
            
            if (_exitNorm > _tolerance){
                //Calculate the new direction
                double r_newT_r_new = _residualNew->dotProduct(_residualNew);
                _beta = r_newT_r_new / r_oldT_r_old;
                //newDirection = r_new + beta * difference
                _residualNew->add(_directionVectorOld, _directionVectorNew , 1.0, _beta);
                
                //Update the old residual and the old difference
                _residualOld->deepCopy(_residualNew);
                _directionVectorOld->deepCopy(_directionVectorNew);
                _xOld->deepCopy(_xNew);
                logs.stopMultipleObservationsTimer("CG Iteration Time");
                logs.setMultipleObservationsLogData("CG Residual Norm", _exitNorm);
                //_printIterationAndNorm(10) ;
            }
            else {
                _xNew->deepCopy(_xOld);
                logs.stopMultipleObservationsTimer("CG Iteration Time");
                logs.setMultipleObservationsLogData("CG Residual Norm", _exitNorm);
                break;
            }
            _iteration++;
            cout << "Iteration: " << _iteration << " Residual Norm: " << _exitNorm << endl;
            //logs.storeAndResetCurrentLogs();
        }
    };
} // LinearAlgebra