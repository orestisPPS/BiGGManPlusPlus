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
    

    class StationaryIterative : public IterativeSolver {

    public:
        explicit StationaryIterative(VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4,
                                     bool throwExceptionOnMaxFailure = true, ParallelizationMethod parallelizationMethod = Wank);
        
        void _iterativeSolution() override;
        
        void _singleThreadSolution() override;
        
        void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows) override;
        
        void _cudaSolution() override;
        
    private:
        unique_ptr<StationaryIterativeCuda> _stationaryIterativeCuda;
        
    };


} // LinearAlgebra

#endif //UNTITLED_STATIONARYITERATIVE_H
