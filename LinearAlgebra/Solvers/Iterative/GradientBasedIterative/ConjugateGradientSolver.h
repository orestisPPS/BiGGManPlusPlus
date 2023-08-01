//
// Created by hal9000 on 7/27/23.
//

#ifndef UNTITLED_CONJUGATEGRADIENTSOLVER_H
#define UNTITLED_CONJUGATEGRADIENTSOLVER_H

#include "../IterativeSolver.h"

namespace LinearAlgebra {

    class ConjugateGradientSolver : public IterativeSolver {
        
    public:
        explicit ConjugateGradientSolver(VectorNormType normType, double tolerance = 1E-9, unsigned maxIterations = 1E4,
                                bool throwExceptionOnMaxFailure = true, ParallelizationMethod parallelizationMethod = Wank);

    protected:
        
        void _initializeVectors() override;
        
        void _iterativeSolution() override;
        
        void _singleThreadSolution() override;

        void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows) override;

        void _cudaSolution() override;
        
        shared_ptr<vector<double>> _residualOld;

        shared_ptr<vector<double>> _residualNew;

        shared_ptr<vector<double>> _directionVectorNew;
        
        shared_ptr<vector<double>> _directionVectorOld;
        
        shared_ptr<vector<double>> _matrixVectorMultiplication;
        
        unique_ptr<double> _alpha;
        
        unique_ptr<double> _beta;
        
    };
} // LinearAlgebra

#endif //UNTITLED_CONJUGATEGRADIENTSOLVER_H