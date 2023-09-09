//
// Created by hal9000 on 8/5/23.
//

#ifndef UNTITLED_ITERATIONQR_H
#define UNTITLED_ITERATIONQR_H

#include "GramSchmidtQR.h"
#include "HouseHolderQR.h"
#include "../../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"

namespace LinearAlgebra {

    class IterationQR {
    public:
        explicit IterationQR(unsigned maxIterations = 20, double exitError = 1E-4, DecompositionType decompositionType = Householder,
                             ParallelizationMethod parallelizationMethod = SingleThread, bool storeOnMatrix = true);
        
        void calculateEigenvalues();

        shared_ptr<vector<double>> getEigenvalues();
        
        shared_ptr<vector<double>> getSortedEigenvalues(bool ascending = false);

        void setMatrix(shared_ptr<Array<double>>&);
        
    private:

        shared_ptr<Array<double>> _matrix;
        
        unsigned _maxIterations;
        
        unsigned _iteration;
        
        double _exitError;

        DecompositionType _decompositionType;

        ParallelizationMethod _parallelizationMethod;
        
        shared_ptr<Array<double>> _matrixCopy;
        
        bool _matrixSet;
        
        bool _storeOnMatrix;
        
        shared_ptr<DecompositionQR> _matrixQRDecomposition;
        
        void _deepCopyMatrix();
    };

} // LinearAlgebra

#endif //UNTITLED_ITERATIONQR_H
