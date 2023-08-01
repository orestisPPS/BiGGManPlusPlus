//
// Created by hal9000 on 7/27/23.
//

#include "ConjugateGradientSolver.h"


namespace LinearAlgebra {
    ConjugateGradientSolver::ConjugateGradientSolver(VectorNormType normType, double tolerance,
                                                     unsigned int maxIterations, bool throwExceptionOnMaxFailure, ParallelizationMethod parallelizationMethod) :
                                                     IterativeSolver(normType, tolerance, maxIterations, throwExceptionOnMaxFailure, parallelizationMethod) {
        _solverName = "Conjugate Gradient";
    }

    void ConjugateGradientSolver::_initializeVectors() {
        auto n = _linearSystem->matrix->numberOfRows();
        _xNew = make_shared<vector<double>>(n);
        _xOld = make_shared<vector<double>>(n);
        _residualOld = make_shared<vector<double>>(n);
        _residualNew = make_shared<vector<double>>(n);
        _directionVectorNew = make_shared<vector<double>>(n);
        _directionVectorOld = make_shared<vector<double>>(n);
        _difference = make_shared<vector<double>>(n);
        _matrixVectorMultiplication = make_shared<vector<double>>(n);
        _vectorsInitialized = true;
    }
    
    void ConjugateGradientSolver::_iterativeSolution() {
        auto start = std::chrono::high_resolution_clock::now();
        unsigned n = _linearSystem->matrix->numberOfRows();
        
        if (_parallelization == Wank) {
            _singleThreadSolution();
        }
        if (_parallelization == vTechKickInYoo) {
            auto numberOfThreads = std::thread::hardware_concurrency();
            _printMultiThreadInitializationText(numberOfThreads);
            while (_iteration < _maxIterations && _exitNorm >= _tolerance) {
                _multiThreadSolution(numberOfThreads, n);
                _exitNorm = _calculateNorm();
                _printIterationAndNorm();
                _iteration++;
            }

        }

        else if (_parallelization == turboVTechKickInYoo) {

            double *d_matrix = _linearSystem->matrix->getArrayPointer();
            double *d_rhs = _linearSystem->rhs->data();
            double *d_xOld = _xOld->data();
            double *d_xNew = _xNew->data();
            int vectorSize = static_cast<int>(_linearSystem->rhs->size());

    

        }

        auto end = std::chrono::high_resolution_clock::now();
        printAnalysisOutcome(_iteration, _exitNorm, start, end);
    }

    void ConjugateGradientSolver::_multiThreadSolution(const unsigned short &availableThreads,
                                                       const unsigned short &numberOfRows) {
        
    }

    void ConjugateGradientSolver::_cudaSolution() {
        IterativeSolver::_cudaSolution();
    }

    void ConjugateGradientSolver::_singleThreadSolution() {
        _printSingleThreadInitializationText();
        auto start = std::chrono::high_resolution_clock::now();
        double alpha = 0.0;
        double beta = 0.0;
        _exitNorm = 1.0;
        //Calculate the initial residual

        auto lol = _linearSystem->matrix->isSymmetric();
        ////_linearSystem->matrix->print(10);
        //setInitialSolution(0);
        //A * x_old
        VectorOperations::matrixVectorMultiplication(_linearSystem->matrix, _xOld, _matrixVectorMultiplication);
        //r_old = b - A * x_old
        VectorOperations::subtract(_linearSystem->rhs, _matrixVectorMultiplication, _residualOld);
        double normInitial = VectorNorm(_residualOld, _normType).value();
        _residualNorms->push_back(normInitial);
        //d_old = r_old
        VectorOperations::deepCopy(_residualOld, _directionVectorOld);

        while (_iteration < _maxIterations) {
            auto matrixTimesDirection = make_shared<vector<double>>(_linearSystem->matrix->numberOfRows(), 0);
            VectorOperations::matrixVectorMultiplication(_linearSystem->matrix, _directionVectorOld, matrixTimesDirection);
            //Calculate the step size
            //alpha = (r_old, r_old)/(difference, A * difference)
            double r_oldT_r_old = VectorOperations::dotProductWithTranspose(_residualOld);
            double direction_oldT_A_direction_old = VectorOperations::dotProduct(_directionVectorOld, matrixTimesDirection);
            alpha = r_oldT_r_old / direction_oldT_A_direction_old;
            
            //x_new = x_old + alpha * difference
            VectorOperations::addScaledVector(_xOld, _directionVectorOld, _xNew, alpha);
            //r_new = r_old - alpha * A * difference
            VectorOperations::subtractScaledVector(_residualOld, matrixTimesDirection, _residualNew, alpha);
            
            //VectorOperations::subtract(_xNew, _xOld, _difference);
            
            //Calculate the norm of the residual
            _exitNorm = VectorNorm(_residualNew, _normType).value() / normInitial;
            //_exitNorm = VectorNorm(_residualNew, _normType).value();
            _residualNorms->push_back(_exitNorm);
            if (_exitNorm > _tolerance){
                //Calculate the new direction
                double r_newT_r_new = VectorOperations::dotProductWithTranspose(_residualNew);
                beta = r_newT_r_new / r_oldT_r_old;
                //newDirection = r_new + beta * difference
                VectorOperations::addScaledVector(_residualNew, _directionVectorOld, _directionVectorNew, beta);
                
                VectorOperations::deepCopy(_residualNew, _residualOld);
                VectorOperations::deepCopy(_directionVectorNew, _directionVectorOld);
                VectorOperations::deepCopy(_xNew, _xOld);
            }
            else {
                VectorOperations::deepCopy(_xNew, _xOld);
                break;
            }
            
            //Update the old residual and the old difference
            _printIterationAndNorm(1) ;
            _iteration++;
        }
        auto end = std::chrono::high_resolution_clock::now();
    };


} // LinearAlgebra