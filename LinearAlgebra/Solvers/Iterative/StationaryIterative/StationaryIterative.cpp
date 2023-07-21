//
// Created by hal9000 on 6/14/23.
//

#include "StationaryIterative.h"

namespace LinearAlgebra {
    StationaryIterative::StationaryIterative(ParallelizationMethod parallelizationMethod, VectorNormType normType, double tolerance, unsigned maxIterations,
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
        double sum = 0.0;
        //Check if the initial solution has been set from the user. If not, it is set to 0.0
        if (!_isInitialized) {
            setInitialSolution(0.0);
        }
        //TODO Check if the matrix is diagonally dominant

        if (_parallelization == vTechKickInYoo) {
            cout<< " " << endl;
            cout << "----------------------------------------" << endl;
            cout << _solverName << " Solver Multi Thread - VTEC KICKED IN YO!" << endl;
            //Find the number of threads available for parallel execution
            unsigned numberOfThreads = std::thread::hardware_concurrency();
            cout << "Total Number of threads available for parallel execution: " << numberOfThreads << endl;
            cout << "Number of threads involved in parallel solution: " << numberOfThreads << endl;

            //Define the available threads (2 less than the number of threads available for parallel execution)
            while (iteration < _maxIterations && norm >= _tolerance) {

                _multiThreadSolution(numberOfThreads, n);
                // Calculate the norm of the _difference
                norm = VectorNorm(_difference, _normType).value();
                // Add the norm to the list of norms
                _residualNorms->push_back(norm);
                if (iteration % 100 == 0) {
                    cout << "Iteration: " << iteration << " - Norm: " << norm << endl;
                }
                iteration++;
            }
        }
        else if (_parallelization == Wank) {
            cout<< " " << endl;
            cout << "----------------------------------------" << endl;
            cout << _solverName << " Solver Single Thread" << endl;
            while (iteration < _maxIterations && norm >= _tolerance) {
                _singleThreadSolution();
                // Calculate the norm of the _difference
                norm = VectorNorm(_difference, _normType).value();
                // Add the norm to the list of norms
                _residualNorms->push_back(norm);
                if (iteration % 100 == 0) {
                    cout << "Iteration: " << iteration << " - Norm: " << norm << endl;
                }
                iteration++;
            }
        }
        else if (_parallelization == turboVTechKickInYoo){
            double* d_matrix = _linearSystem->matrix->getArrayPointer();
            auto matrixSize = _linearSystem->matrix->size();
            cudaMalloc(&d_matrix, matrixSize * sizeof(double));
            
            double* d_rhs = _linearSystem->rhs->data();
            auto rhsSize = _linearSystem->rhs->size();
            cudaMalloc(&d_rhs, rhsSize * sizeof(double));
            
            double* d_solution = _linearSystem->solution->data();
            auto solutionSize = _linearSystem->solution->size();
            cudaMalloc(&d_solution, solutionSize * sizeof(double));
            
            cout<< " allocated first gpu stuff boi" << endl;
        }
        // If the maximum number of iterations is reached and the user has set the flag to throw an exception, throw an exception
        if (iteration == _maxIterations && _throwExceptionOnMaxFailure) {
            throw std::runtime_error("Maximum number of iterations reached.");
        }
        auto end = std::chrono::high_resolution_clock::now();

        bool isInMicroSeconds = false;
        auto _elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (_elapsedTime == 0) {
            _elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            isInMicroSeconds = true;
        }
        if (isInMicroSeconds) {
            cout << "Elapsed time: " << _elapsedTime << " μs" << " Iterations : " << iteration << " Exit norm : " << norm << endl;
            cout << "----------------------------------------" << endl;
        }
        else {
            cout << "Elapsed time: " << _elapsedTime << " ms" << " Iterations : " << iteration << " Exit norm : " << norm << endl;
            cout << "----------------------------------------" << endl;
        }
    }

    void StationaryIterative::_multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows) {

        //Initiate the thread pool map
        map<unsigned, thread> threadPool = map <unsigned, thread>();
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
        for (auto& thread : threadPool) {
            thread.second.join();
        }

    }

    void StationaryIterative::_singleThreadSolution() {
        _threadJob(0, _linearSystem->matrix->numberOfRows());
    }

    void StationaryIterative::_threadJob(unsigned start, unsigned end) {

    }
} // LinearAlgebra