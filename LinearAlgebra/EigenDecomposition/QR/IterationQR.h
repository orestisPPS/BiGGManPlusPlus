//
// Created by hal9000 on 8/5/23.
//

#ifndef UNTITLED_ITERATIONQR_H
#define UNTITLED_ITERATIONQR_H

#include "GramSchmidtQR.h"
#include "HouseHolderQR.h"

namespace LinearAlgebra {

    class IterationQR {
    private:
        IterationQR(unsigned maxIterations = 20, double exitError = 1E-4, DecompositionType decompositionType = Householder, ParallelizationMethod parallelizationMethod = Wank, bool storeOnMatrix = false);
        
        void calculateEigenValues();

        shared_ptr<vector<double>> getEigenvalues();

        void setMatrix(shared_ptr<Array<double>>&);
        
        shared_ptr<Array<double>> getMatrix();

    private:

        shared_ptr<Array<double>> _matrix;
        
        shared_ptr<Array<double>> _Q;
        
        shared_ptr<Array<double>> _R;
        
        unsigned _maxIterations;
        
        unsigned _iteration;
        
        double _exitError;

        DecompositionType _decompositionType;

        ParallelizationMethod _parallelizationMethod;
        
        shared_ptr<Array<double>> _matrixCopy;
        
        
        void _deepCopyMatrix();
        
        bool _matrixSet;
        
        bool _storeOnMatrix;
        
        shared_ptr<DecompositionQR> _matrixQRDecomposition;
    };

} // LinearAlgebra

#endif //UNTITLED_ITERATIONQR_H
