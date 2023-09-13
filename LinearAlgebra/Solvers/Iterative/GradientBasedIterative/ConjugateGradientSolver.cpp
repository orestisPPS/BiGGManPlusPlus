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

    void ConjugateGradientSolver::_performMethodIteration() {
        _printInitializationText();
        auto start = std::chrono::high_resolution_clock::now();
        double alpha = 0.0;
        double beta = 0.0;
        _exitNorm = 1.0;
        //Calculate the initial residual
        //setInitialSolution(0);
        //A * x_old
        _linearSystem->matrix->multiplyVector(_xOld, _matrixVectorMultiplication);
        //r_old = b - A * x_old
        _linearSystem->rhs->subtract(_matrixVectorMultiplication, _residualOld);
        double normInitial = _residualOld->norm(L2);
        _residualNorms->push_back(normInitial);
        //d_old = r_old
        *_directionVectorOld = *_residualOld;

        while (_iteration < _maxIterations) {
            _matrixVectorMultiplication->fill(0.0);
            _linearSystem->matrix->multiplyVector(_directionVectorOld, _matrixVectorMultiplication);
            //Calculate the step size
            //alpha = (r_old, r_old)/(difference, A * difference)
            
            double r_oldT_r_old = _residualOld->dotProduct(_residualOld);
            double direction_oldT_A_direction_old = _directionVectorOld->dotProduct(_matrixVectorMultiplication);
            alpha = r_oldT_r_old / direction_oldT_A_direction_old;
            
            //x_new = x_old + alpha * difference
            //_xOld->add(_directionVectorOld, _xNew, 1.0, alpha);
            _xOld->add(_directionVectorOld, _xNew, 1.0, alpha);
            //r_new = r_old - alpha * A * difference
            _residualOld->subtract(_matrixVectorMultiplication, _residualNew, 1.0, alpha);
            
            //VectorOperations::subtract(_xNew, _xOld, _difference);
            
            //Calculate the norm of the residual
            _exitNorm = _residualNew->norm(_normType);// / normInitial;
            //_exitNorm = VectorNorm(_residualNew, _normType).value();
            _residualNorms->push_back(_exitNorm);
            if (_exitNorm > _tolerance){
                //Calculate the new direction
                double r_newT_r_new = _residualNew->dotProduct(_residualNew);
                beta = r_newT_r_new / r_oldT_r_old;
                //newDirection = r_new + beta * difference
                _residualNew->add(_directionVectorOld, _directionVectorNew , 1.0, beta);
                
                //Update the old residual and the old difference
                *_residualOld = *_residualNew;
                *_directionVectorOld = *_directionVectorNew;
                *_xOld = *_xNew;
                //_printIterationAndNorm(10) ;
            }
            else {
                *_xOld = *_xNew;
                break;
            }
            
            //Update the old residual and the old difference
            //_printIterationAndNorm(1) ;
            _iteration++;
        }
        auto end = std::chrono::high_resolution_clock::now();
    };
} // LinearAlgebra