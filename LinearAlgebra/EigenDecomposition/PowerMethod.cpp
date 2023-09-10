/*
//
// Created by hal9000 on 8/4/23.
//

#include "PowerMethod.h"

namespace LinearAlgebra {
    template<typename T>
    PowerMethod<T>::PowerMethod(shared_ptr<NumericalMatrix<T>> matrix, unsigned maxIterations = 1E4, double tolerance = 1E-5,
                             VectorNormType normType = L2) {
        _numberOfEigenvalues = numberOfEigenvalues;
        _normType = normType;
        _maxIterations = lanczosIterations;
        _tolerance = tolerance;
        _parallelizationMethod = parallelizationMethod;
        _iteration = 0;
        _exitNorm = 10;
        _vectorsInitialized = false;
        _matrixSet = false;
    }

    void PowerMethod::calculateDominantEigenValue() {
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
       
    

    void PowerMethod::_singleThreadSolution(){
        _printSingleThreadInitializationText();

*/
/*
        for (unsigned i = 0; i < _matrix->numberOfRows(); i++){
            for (int j = 0; j < _matrix->numberOfRows() ; ++j) {
                _matrix->at(i,j) = _matrix->at(i,j) * (-1.0);
            }
        }
*//*


        while (_iteration < _maxIterations && _exitNorm > _tolerance){
            VectorOperations::matrixVectorMultiplication(_matrix, _vectorOld, _vectorNew);
            auto norm = VectorNorm(_vectorNew, _normType).value();
            VectorOperations::scale(_vectorNew, 1 / norm);
            VectorOperations::deepCopy(_vectorNew, _vectorOld);
            _iteration++;
        }
        
        auto eig = VectorOperations::dotProduct(_vectorOld, _vectorNew);
        cout<<eig<<endl;

    }
    

    void PowerMethod::setMatrix(const shared_ptr<NumericalMatrix<double>>& matrix) {
        _matrix = matrix;
        _matrixSet = true;
    }

    void PowerMethod::_initializeVectors() {
        if (_matrixSet) {
            auto n = _matrix->numberOfRows();

            _vectorNew = make_shared<NumericalVector<double>>(n, 0);
            _vectorOld = make_shared<NumericalVector<double>>(n, 0);
            _difference = make_shared<NumericalVector<double>>(n, 0);
            // Random number generation setup using C++'s <random> library
            // Mersenne Twister generator
            std::mt19937 generator;
            // Seed with a device-dependent random number
            generator.seed(std::random_device()());
            // Uniform distribution between -1 and 1
            std::uniform_real_distribution<double> distribution(0, 1); 
            //std::uniform_real_distribution<double> distribution(-40, 0.1);

            //Fill _lanczosVectorNew with random numbers and normalize it
            for (auto &component : *_vectorOld) {
                component = distribution(generator);
            }
            // Normalize the vector
            VectorOperations::normalize(_vectorOld);
            _vectorsInitialized = true;

        } else {
            throw runtime_error("NumericalMatrix not set");
        }
    }

    void PowerMethod::_printSingleThreadInitializationText() {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << "Power Method" << " Eigen-decomposition Single Thread - no vtec yo :(" << endl;
    }

    void PowerMethod::_printMultiThreadInitializationText(unsigned short numberOfThreads) {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << "Power Method" << " Eigen-decomposition Multi Thread - VTEC KICKED IN YO!" << endl;
        //Find the number of threads available for parallel execution
        cout << "Total Number of threads available for parallel execution: " << numberOfThreads << endl;
        cout << "Number of threads involved in parallel solution: " << numberOfThreads << endl;
    }

    void PowerMethod::_printCUDAInitializationText() {

    }

    void PowerMethod::_printIterationAndNorm(unsigned displayFrequency) const {
        if (_iteration % displayFrequency == 0)
            cout << "Iteration: " << _iteration << " - Norm: " << _exitNorm << endl;

    }


    void PowerMethod::printAnalysisOutcome(unsigned totalIterations, double exitNorm,  std::chrono::high_resolution_clock::time_point startTime,
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

} // LinearAlgebra*/
