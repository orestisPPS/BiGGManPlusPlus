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
        explicit StationaryIterative(double tolerance, unsigned maxIterations, VectorNormType normType,
                                     unsigned userDefinedThreads, bool printOutput, bool throwExceptionOnMaxFailure);

        void _initializeVectors() override;
        
        void _iterativeSolution() override;
        
        void _performMethodIteration() override;
        
        void _cudaSolution() override;
        
        
        
    protected:

    private:
        unique_ptr<StationaryIterativeCuda> _stationaryIterativeCuda;
        
    };


} // LinearAlgebra

#endif //UNTITLED_STATIONARYITERATIVE_H
