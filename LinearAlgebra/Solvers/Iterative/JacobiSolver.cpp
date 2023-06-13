//
// Created by hal9000 on 6/13/23.
//

#include "JacobiSolver.h"

namespace LinearAlgebra {

    JacobiSolver::JacobiSolver(bool vTechKickInYoo, VectorNormType normType, double tolerance, unsigned maxIterations,
                               bool throwExceptionOnMaxFailure) :
            IterativeSolver(normType, tolerance, maxIterations, throwExceptionOnMaxFailure) {
        _vTechKickInYoo = vTechKickInYoo;
        _residualNorms = make_shared<list<double>>();
    }

    void JacobiSolver::_iterativeSolution() {
        auto start = std::chrono::high_resolution_clock::now();
        double n = _linearSystem->matrix->numberOfRows();
        unsigned short iteration = 0;
        double norm = 1.0;
        double sum = 0.0;
        //Check if the initial solution has been set from the user. If not, it is set to 0.0
        if (!_isInitialized) {
            setInitialSolution(0.0);
        }
        //TODO Check if the matrix is diagonally dominant
        
        if (_vTechKickInYoo) {
            cout << "----------------------------------------" << endl;
            cout << "Jacobi Solver Multi Thread" << endl;
            
            while (iteration < _maxIterations && norm >= _tolerance) {
                multiThreadSolution();
                // Calculate the norm of the _difference
                norm = VectorNorm(_difference, _normType).value();
                // Add the norm to the list of norms
                _residualNorms->push_back(norm);
                iteration++;
            }
        }
        else {
            cout << "----------------------------------------" << endl;
            cout << "Jacobi Solver Single Thread" << endl;
            while (iteration < _maxIterations && norm >= _tolerance) {
                singleThreadSolution();
                // Calculate the norm of the _difference
                norm = VectorNorm(_difference, _normType).value();
                // Add the norm to the list of norms
                _residualNorms->push_back(norm);
                iteration++;
            }
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
    
    void JacobiSolver::multiThreadSolution() {
        double n = _linearSystem->matrix->numberOfRows();
        unsigned short iteration = 0;
        double norm = 0.0;
        double sum = 0.0;
        
        //Find the number of threads available for parallel execution
        unsigned numberOfThreads = std::thread::hardware_concurrency();
        cout << "Number of threads available for parallel execution: " << numberOfThreads << endl;
        
        //Define the available threads (2 less than the number of threads available for parallel execution)
        unsigned numberOfThreadsToUse = numberOfThreads - 2;
        cout << "Number of threads to use: " << numberOfThreadsToUse << endl;
        
        //Initiate the thread vector
        vector<thread> threads(numberOfThreads - 2);

        for (int i = 0; i < n ; ++i) {
            
        }
        
        
        // Define the thread task for each row
        auto threadTask = [this, n](unsigned startRow, unsigned endRow) {
            // Iterate over the rows assigned to the thread
            for (unsigned int i = startRow; i < endRow; ++i) {
                // Calculate the sum of the row
                double sum = 0.0;
                for (unsigned int j = 0; j < n; ++j) {
                    if (i != j) {
                        sum += _linearSystem->matrix->at(i, j) * _xOld->at(j);
                    }
                }
            }
        };

        // Calculate the number of rows to assign to each thread
        unsigned int rowsPerThread = n / numberOfThreadsToUse;
        unsigned int startRow = 0;
        unsigned int endRow = rowsPerThread;

        // Launch the threads and assign the work
        for (unsigned int i = 0; i < numberOfThreadsToUse; ++i) {
            // For the last thread, adjust the endRow to cover the remaining rows
            if (i == numberOfThreadsToUse - 1) {
                endRow = n;
            }

            // Launch the thread and assign the task
            threads[i] = thread(threadTask, startRow, endRow);

            // Update the startRow and endRow for the next thread
            startRow = endRow;
            endRow += rowsPerThread;
        }

        // Wait for all the threads to finish
        for (auto& thread : threads) {
            thread.join();
        }
    }

    void JacobiSolver::singleThreadSolution() {

        double n = _linearSystem->matrix->numberOfRows();
        unsigned short iteration = 0;
        double norm = 0.0;
        double sum = 0.0;

        for (unsigned i = 0; i < n; i++) {
            //Zero difference and sum
            sum = 0.0;
            _difference->assign(unsigned(n), 0.0);
            for (unsigned j = 0; j < n; j++) {
                if (i != j) {
                    // sum = sum + A_ij * xOld_j
                    sum += _linearSystem->matrix->at(i, j) * _xOld->at(j);
                }
            }
            // xNew_i = (b_i - sum) / A_ii
            _xNew->at(i) = (_linearSystem->RHS->at(i) - sum) / _linearSystem->matrix->at(i, i);
            // Calculate the _difference
            _difference->at(i) = _xNew->at(i) - _xOld->at(i);
            //Replace the old solution with the new one
            _xOld->at(i) = _xNew->at(i);
        }
    }

} // LinearAlgebra