//
// Created by hal9000 on 8/5/23.
//

#ifndef UNTITLED_DECOMPOSITIONQR_H
#define UNTITLED_DECOMPOSITIONQR_H

#include "../../Array/Array.h"
#include "../../ParallelizationMethods.h"
#include "../../Operations/VectorOperations.h"
#include "../../Norms/VectorNorm.h"
#include "../../Operations/MultiThreadVectorOperations.h"

namespace LinearAlgebra {
    
    enum DecompositionType {
        GramSchmidt,
        Householder
    };
    
    class DecompositionQR {

    public:
        explicit DecompositionQR(bool returnQ = true, ParallelizationMethod parallelizationMethod = Wank, bool storeOnMatrix = false);
        
        void setMatrix(shared_ptr<Array<double>>&);

        virtual shared_ptr<Array<double>> getQ();

        virtual shared_ptr<Array<double>> getR();

        virtual shared_ptr<Array<double>> getMatrix();
        
        
        void decompose();
        
    protected:
        shared_ptr<Array<double>> _matrix;
        
        shared_ptr<Array<double>> _Q;
        
        shared_ptr<Array<double>> _R;
        
        DecompositionType _decompositionType;
        
        bool _returnQ;
        
        bool _matrixSet;
        
        ParallelizationMethod _parallelizationMethod;
        
        bool _storeOnMatrix;
        

        virtual void _singleThreadDecomposition();
        
        virtual void _multiThreadDecomposition();
        
        virtual void _CUDADecomposition();
    };

} // LinearAlgebra

#endif //UNTITLED_DECOMPOSITIONQR_H
