//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_STATIONARYITERATIVE_H
#define UNTITLED_STATIONARYITERATIVE_H

#include <thread>
#include "../IterativeSolver.h"

namespace LinearAlgebra {
    
    class StationaryIterative : public IterativeSolver {

    public:
        StationaryIterative(bool vTechKickInYoo, VectorNormType normType, double tolerance = 1E-9, unsigned maxIterations = 1E4, bool throwExceptionOnMaxFailure = true);

    protected:
        void _iterativeSolution() override;

        void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows);

        virtual void _singleThreadSolution();
        
        //Thread job. Parallel for each row. Changes for each solver.
        virtual void _threadJob(unsigned start, unsigned end);

        bool _vTechKickInYoo;
        
        string _solverName;
        
    };


} // LinearAlgebra

#endif //UNTITLED_STATIONARYITERATIVE_H
