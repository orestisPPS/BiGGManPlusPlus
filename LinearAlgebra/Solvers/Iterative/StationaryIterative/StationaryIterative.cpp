//
// Created by hal9000 on 6/14/23.
//

#include "StationaryIterative.h"

namespace LinearAlgebra {
    StationaryIterative::StationaryIterative(ParallelizationMethod parallelizationMethod, VectorNormType normType,
                                             double tolerance, unsigned maxIterations,
                                             bool throwExceptionOnMaxFailure) :
            IterativeSolver(normType, tolerance, maxIterations, throwExceptionOnMaxFailure) {
        _parallelization = parallelizationMethod;
        _residualNorms = make_shared<list<double>>();
    }

    void StationaryIterative::_iterativeSolution() {
        auto start = std::chrono::high_resolution_clock::now();
        unsigned n = _linearSystem->matrix->numberOfRows();
        unsigned short iteration = 0;
        double norm = 1.0;
        
        //Check if the initial solution has been set from the user. If not, it is initialized to 0.0
        if (!_isInitialized)
            setInitialSolution(0.0);
        
        if (_parallelization == Wank) {
            _printSingleThreadInitializationText();
            while (iteration < _maxIterations && norm >= _tolerance) {
                _singleThreadSolution();
                norm = _calculateNorm();
                _printIterationAndNorm(iteration, norm);
                iteration++;
            }
        }
        if (_parallelization == vTechKickInYoo) {

            //Define the available threads (2 less than the number of threads available for parallel execution)
            unsigned numberOfThreads = std::thread::hardware_concurrency();
            _printMultiThreadInitializationText(numberOfThreads);
            while (iteration < _maxIterations && norm >= _tolerance) {
                _multiThreadSolution(numberOfThreads, n);
                norm = _calculateNorm();
                _printIterationAndNorm(iteration, norm);
                iteration++;
            }
        }

        else if (_parallelization == turboVTechKickInYoo) {

/*

            }*/
            //cuda file with all gpu utility as extern here 

            double *d_matrix = _linearSystem->matrix->getArrayPointer();
            double *d_rhs = _linearSystem->rhs->data();
            double *d_xOld = _xOld->data();
            double *d_xNew = _xNew->data();
            double *d_difference = _difference->data();
            int vectorSize = static_cast<int>(_linearSystem->rhs->size());

            _stationaryIterativeCuda = make_unique<StationaryIterativeCuda>(
                    d_matrix, d_rhs, d_xOld, d_xNew, d_difference, vectorSize, 256);

            while (iteration < _maxIterations && norm >= _tolerance) {

                _stationaryIterativeCuda->performGaussSeidelIteration();
                _stationaryIterativeCuda->getDifferenceVector(_difference->data());
                norm = VectorNorm(_difference, _normType).value();
                //norm = _stationaryIterativeCuda->getNorm();
                // Add the norm to the list of norms
                _residualNorms->push_back(norm);

                if (iteration % 100 == 0) {
                    cout << "Iteration: " << iteration << " - Norm: " << norm << endl;
                }

                iteration++;
            }



            _stationaryIterativeCuda->getSolutionVector(_xNew->data());
            _stationaryIterativeCuda.reset();

        }

        auto end = std::chrono::high_resolution_clock::now();
        printAnalysisOutcome(iteration, norm, start, end);
        
    }

    void StationaryIterative::_multiThreadSolution(const unsigned short &availableThreads,
                                                   const unsigned short &numberOfRows) {

        //Initiate the thread pool map
        map<unsigned, thread> threadPool = map<unsigned, thread>();
        for (unsigned int i = 0; i < availableThreads; ++i) {
            threadPool.insert(pair<unsigned, thread>(i, thread()));
        }

        // Calculate the number of rows to assign to each thread
        unsigned rowsPerThread = numberOfRows / availableThreads;
        unsigned lastRows = numberOfRows % availableThreads;

        // Launch the threads and assign the work
        unsigned startRow = 0;
        unsigned endRow = 0;
        for (unsigned i = 0; i < availableThreads; ++i) {
            endRow = startRow + rowsPerThread;
            if (i == availableThreads - 1) {
                endRow = endRow + lastRows;
            }
            threadPool[i] = thread(&StationaryIterative::_threadJob, this, startRow, endRow);
            startRow = endRow;
        }
        // Wait for all the threads to finish
        for (auto &thread: threadPool) {
            thread.second.join();
        }

    }

    void StationaryIterative::_singleThreadSolution() {
        _threadJob(0, _linearSystem->matrix->numberOfRows());
    }

    void StationaryIterative::_threadJob(unsigned start, unsigned end) {

    }

    void StationaryIterative::_printSingleThreadInitializationText() {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << _solverName << " Solver Single Thread - no vtec yo :(" << endl;
    }

    void StationaryIterative::_printMultiThreadInitializationText(unsigned short numberOfThreads) {
        cout << " " << endl;
        cout << "----------------------------------------" << endl;
        cout << _solverName << " Solver Multi Thread - VTEC KICKED IN YO!" << endl;
        //Find the number of threads available for parallel execution
        cout << "Total Number of threads available for parallel execution: " << numberOfThreads << endl;
        cout << "Number of threads involved in parallel solution: " << numberOfThreads << endl;
    }

    void StationaryIterative::_printCUDAInitializationText() {

    }

    void StationaryIterative::_printIterationAndNorm(unsigned int iteration, double norm) {
        if (iteration % 100 == 0)
            cout << "Iteration: " << iteration << " - Norm: " << norm << endl;

    }

    double StationaryIterative::_calculateNorm() {
        double norm = VectorNorm(_difference, _normType).value();
        _residualNorms->push_back(norm);
        return norm;
    }

    void StationaryIterative::printAnalysisOutcome(unsigned totalIterations, double exitNorm,  std::chrono::high_resolution_clock::time_point startTime,
                                                   std::chrono::high_resolution_clock::time_point finishTime){
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
}
// LinearAlgebra