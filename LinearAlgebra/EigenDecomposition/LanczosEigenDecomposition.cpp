//
// Created by hal9000 on 8/3/23.
//
#include <random>
#include "LanczosEigenDecomposition.h"

using namespace LinearAlgebra;

namespace LinearAlgebra {
    
    LanczosEigenDecomposition::LanczosEigenDecomposition(unsigned short numberOfEigenvalues, VectorNormType normType,
                                                         unsigned short maxIterations, double tolerance,
                                                         ParallelizationMethod parallelizationMethod) {
        _numberOfEigenvalues = numberOfEigenvalues;
        _normType = normType;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _parallelizationMethod = parallelizationMethod;
        _iteration = 0;
        _exitNorm = 1;
        _vectorsInitialized = false;
        _matrixSet = false;
    }
    
    void LanczosEigenDecomposition::calculateEigenvalues() {
        auto start = std::chrono::high_resolution_clock::now();
        _initializeVectors();
        switch (_parallelizationMethod) {
            case Wank:
                _singleThreadSolution();
                break;
            case vTechKickInYoo:
                //();
                break;
            case turboVTechKickInYoo:
                //_cudaSolution();
                break;
        }
        auto end = std::chrono::high_resolution_clock::now();
        printAnalysisOutcome(_iteration, _exitNorm, start, end);
    }
    
    void LanczosEigenDecomposition::_singleThreadSolution(){
        _printSingleThreadInitializationText();
        _alpha = 0.0;
        _beta = 0.0;
        
        
        while (_iteration < 100){
            auto workingVector = make_shared<vector<double>>(_matrix->numberOfRows());
            VectorOperations::matrixVectorMultiplication(_matrix, _lanczosVectorOld, workingVector);
            //Calculate α
            _alpha = VectorOperations::dotProduct(_lanczosVectorOld, workingVector);
            //Orthogonalize
            _singleThreadOrthogonalization(workingVector);
            
            _beta = VectorNorm(workingVector, L2).value();
            VectorOperations::scale(workingVector, 1.0 / _beta);
            VectorOperations::deepCopy(workingVector, _lanczosVectorNew, 1 / _beta);
            _lanczosVectors->insert(pair<unsigned, shared_ptr<vector<double>>>(_iteration, std::move(workingVector)));

            (*_T_diagonal)[_iteration] = _alpha;
            if (_iteration > 0) 
                (*_T_subDiagonal)[_iteration - 1] = _beta;
            
            VectorOperations::deepCopy(_lanczosVectorNew, _lanczosVectorOld);
            auto dot = VectorOperations::dotProduct(_lanczosVectorNew, _lanczosVectorOld);
            
            _iteration++;
        }
        cout<<"mitsotakis"<<endl;

    }
    
    void LanczosEigenDecomposition::_singleThreadOrthogonalization(const shared_ptr<vector<double>> &vectorToOrthogonalize) {
        for (unsigned i = 0; i < _matrix->numberOfRows(); i++) {
            (*vectorToOrthogonalize)[i] -= _beta * (*_lanczosVectorOld)[i] + _alpha * (*_lanczosVectorNew)[i];
        }
    }
            
    
    void LanczosEigenDecomposition::setMatrix(const shared_ptr<Array<double>>& matrix) {
        _matrix = matrix;
        _matrixSet = true;
    }
    
    void LanczosEigenDecomposition::_initializeVectors() {
        if (_matrixSet) {
            auto n = _matrix->numberOfRows();
            _lanczosVectors = make_shared<map<unsigned, shared_ptr<vector<double>>>>();            
            _lanczosVectorOld= make_shared<vector<double>>(n);
            // Random number generation setup using C++'s <random> library
            // Mersenne Twister generator
            std::mt19937 generator; 
            // Seed with a device-dependent random number
            generator.seed(std::random_device()()); 
            // Uniform distribution between -1 and 1
            std::uniform_real_distribution<double> distribution(-1.0, 1.0); 

            //Fill _lanczosVectorNew with random numbers and normalize it
            for (auto &component : *_lanczosVectorOld) {
                component = distribution(generator);
            }
            // Normalize the vector
            VectorOperations::normalize(_lanczosVectorOld);
            
            _lanczosVectorNew= make_shared<vector<double>>(n, 0);
            
            _T_diagonal = make_shared<vector<double>>(_maxIterations, 0);
            _T_subDiagonal = make_shared<vector<double>>(_maxIterations - 1, 0);
            
            _vectorsInitialized = true;
            
        } else {
            throw runtime_error("Matrix not set");
        }
    }

    void LanczosEigenDecomposition::_printSingleThreadInitializationText() {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << "Lanczos" << " Eigen-decomposition Single Thread - no vtec yo :(" << endl;
    }

    void LanczosEigenDecomposition::_printMultiThreadInitializationText(unsigned short numberOfThreads) {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << "Lanczos" << " Eigen-decomposition Multi Thread - VTEC KICKED IN YO!" << endl;
        //Find the number of threads available for parallel execution
        cout << "Total Number of threads available for parallel execution: " << numberOfThreads << endl;
        cout << "Number of threads involved in parallel solution: " << numberOfThreads << endl;
    }

    void LanczosEigenDecomposition::_printCUDAInitializationText() {

    }

    void LanczosEigenDecomposition::_printIterationAndNorm(unsigned displayFrequency) const {
        if (_iteration % displayFrequency == 0)
            cout << "Iteration: " << _iteration << " - Norm: " << _exitNorm << endl;

    }

    double LanczosEigenDecomposition::_calculateNorm() {
/*        double norm = VectorNorm(_difference, _normType).value();
        _residualNorms->push_back(norm);
        return norm;*/
    }

    void LanczosEigenDecomposition::printAnalysisOutcome(unsigned totalIterations, double exitNorm,  std::chrono::high_resolution_clock::time_point startTime,
                                               std::chrono::high_resolution_clock::time_point finishTime) const{
        bool isInMicroSeconds = false;
        auto _elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(finishTime - startTime).count();
        if (_elapsedTime == 0) {
            _elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(finishTime - startTime).count();
            isInMicroSeconds = true;
        }
        if (isInMicroSeconds) {
            if (exitNorm <= _tolerance)
                cout << "Convergence Achieved!" << endl;
            else
                cout << "Convergence Failed!" << endl;

            cout << "Elapsed time: " << _elapsedTime << " μs" << " Iterations : " << totalIterations << " Exit norm : " << exitNorm << endl;
            cout << "----------------------------------------" << endl;
        } else {

            if (exitNorm <= _tolerance)
                cout << "Convergence Achieved!" << endl;
            else
                cout << "Convergence Failed!" << endl;

            cout << "Elapsed time: " << _elapsedTime << " ms" << " Iterations : " << totalIterations << " Exit norm : " << exitNorm << endl;
            cout << "----------------------------------------" << endl;
        }

    }


} // LinearAlgebra