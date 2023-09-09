//
// Created by hal9000 on 8/4/23.
//

#ifndef UNTITLED_POWERMETHOD_H
#define UNTITLED_POWERMETHOD_H

#include "IEigenvalueDecomposition.h"
#include "../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"

namespace LinearAlgebra {


    template <typename T>
    class PowerMethod {
    public:

        explicit PowerMethod(shared_ptr<NumericalMatrix<T>> matrix, double tolerance, unsigned maxIterations, VectorNormType normType) :
                matrix(matrix),
                _tolerance(tolerance),
                _maxIterations(maxIterations),
                _normType(normType) {
            _iteration = 0;
            _exitNorm = 10;
            _vectorNew = make_unique<NumericalVector<double>>(matrix->numberOfRows(), 0, matrix->dataStorage->getAvailableThreads());
            _vectorOld = make_unique<NumericalVector<double>>(matrix->numberOfRows(), 0, matrix->dataStorage->getAvailableThreads());
            _vectorOld->fillRandom(0, 1);
            _vectorOld->normalize();
        }

        void calculateDominantEigenValue(){
            matrix->multiplyVector(_vectorOld, _vectorNew);
            //TOOD: check norm type there are two enums
            _exitNorm = _vectorNew->norm(L22);
        }
        
    private:

        void _cpuSolution();
        

        shared_ptr<NumericalMatrix<T>> matrix;
        
        unique_ptr<NumericalVector<double>> _vectorNew;

        unique_ptr<NumericalVector<double>> _vectorOld;
        
        VectorNormType _normType;

        double _tolerance;

        unsigned _iteration;

        unsigned _maxIterations;

        double _exitNorm;
        
        static void _printSingleThreadInitializationText();

        static void _printMultiThreadInitializationText(unsigned short numberOfThreads);

        void _printCUDAInitializationText();

        void _printIterationAndNorm(unsigned displayFrequency = 100) const;
        
        void printAnalysisOutcome(unsigned totalIterations, double exitNorm, std::chrono::high_resolution_clock::time_point startTime,
                                  std::chrono::high_resolution_clock::time_point finishTime) const;
    };

} // LinearAlgebra

#endif //UNTITLED_POWERMETHOD_H


/*
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

}*/
