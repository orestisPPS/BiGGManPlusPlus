//
// Created by hal9000 on 7/27/23.
//

#ifndef UNTITLED_CONJUGATEGRADIENTSOLVER_H
#define UNTITLED_CONJUGATEGRADIENTSOLVER_H

#include "../IterativeSolver.h"

namespace LinearAlgebra {

    class ConjugateGradientSolver : public IterativeSolver {
        
    public:
        explicit ConjugateGradientSolver(double tolerance = 1E-5, unsigned maxIterations = 1E4, VectorNormType normType = L2,
                                         unsigned userDefinedThreads = 0, bool printOutput = true, bool throwExceptionOnMaxFailure = true);

    protected:
        
        void _initializeVectors() override;
        
        void _performMethodSolution() override;
        
        void _cudaSolution() override;
        
        shared_ptr<NumericalVector<double>> _residualOld;

        shared_ptr<NumericalVector<double>> _residualNew;

        shared_ptr<NumericalVector<double>> _directionVectorNew;
        
        shared_ptr<NumericalVector<double>> _directionVectorOld;
        
        shared_ptr<NumericalVector<double>> _matrixVectorMultiplication;
        
        unique_ptr<double> _alpha;
        
        unique_ptr<double> _beta;
        
    };
} // LinearAlgebra

#endif //UNTITLED_CONJUGATEGRADIENTSOLVER_H