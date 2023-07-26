//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_STATIONARYITERATIVE_H
#define UNTITLED_STATIONARYITERATIVE_H

#include <thread>
#include <cuda_runtime.h>
#include "../IterativeSolver.h"
#include "StationaryIterrativeCuda/StationaryIterativeCuda.cuh"

namespace LinearAlgebra {
    
    enum ParallelizationMethod{
        //Multi-thread solution
        vTechKickInYoo,
        //INSSSSSSSSSSSSANE GPU GAINS
        turboVTechKickInYoo,
        //:( Single thread 
        Wank
    };
    class StationaryIterative : public IterativeSolver {

    public:
        StationaryIterative(ParallelizationMethod parallelizationMethod, VectorNormType normType, double tolerance = 1E-9, unsigned maxIterations = 1E4, bool throwExceptionOnMaxFailure = true);
        
    protected:
        void _iterativeSolution() override;

        void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows);
        

        virtual void _singleThreadSolution();
        
        //Thread job. Parallel for each row. Changes for each solver.
        virtual void _threadJob(unsigned start, unsigned end);

        ParallelizationMethod _parallelization;
        
        unique_ptr<StationaryIterativeCuda> _stationaryIterativeCuda;
        
        string _solverName;
        
    private:
        void _printSingleThreadInitializationText();
        void _printMultiThreadInitializationText(unsigned short numberOfThreads);
        void _printCUDAInitializationText();
        static void _printIterationAndNorm(unsigned iteration, double norm);
        double _calculateNorm();
        void printAnalysisOutcome(unsigned totalIterations, double exitNorm, std::chrono::high_resolution_clock::time_point startTime,
                                  std::chrono::high_resolution_clock::time_point finishTime);
        
    };


} // LinearAlgebra

#endif //UNTITLED_STATIONARYITERATIVE_H
