//
// Created by hal9000 on 8/5/23.
//

#ifndef UNTITLED_DECOMPOSITIONQR_H
#define UNTITLED_DECOMPOSITIONQR_H

#include "../../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../Operations/MultiThreadVectorOperations.h"

namespace LinearAlgebra {
    
    enum DecompositionType {
        GramSchmidt,
        Householder
    };
    
    class DecompositionQR {

    public:
        explicit DecompositionQR(bool returnQ = true, ParallelizationMethod parallelizationMethod = SingleThread, bool storeOnMatrix = false);
        
        void setMatrix(shared_ptr<NumericalMatrix<double>>&);
        
        virtual const shared_ptr<NumericalMatrix<double>> & getQ();

        virtual const shared_ptr<NumericalMatrix<double>> &  getR();
        
        void getRQ(shared_ptr<NumericalMatrix<double>>& result);
        
        void decompose();
        
    protected:

        void _getRQSingleThread(shared_ptr<NumericalMatrix<double>>& result);

        void _getRQMultithread(shared_ptr<NumericalMatrix<double>>& result);

        void _getRQCuda(shared_ptr<NumericalMatrix<double>>& result);
        
        shared_ptr<NumericalMatrix<double>> _matrix;
        
        shared_ptr<NumericalMatrix<double>> _Q;
        
        shared_ptr<NumericalMatrix<double>> _R;
        
        DecompositionType _decompositionType;
        
        bool _returnQ;
        
        ParallelizationMethod _parallelizationMethod;
        
        bool _storeOnMatrix;
        
        bool _matrixSet;


        void _deepCopyMatrixIntoR();

        void _initializeArrays();
        
        virtual void _singleThreadDecomposition();
        
        virtual void _multiThreadDecomposition();
        
        virtual void _CUDADecomposition();


    };

} // LinearAlgebra

#endif //UNTITLED_DECOMPOSITIONQR_H
