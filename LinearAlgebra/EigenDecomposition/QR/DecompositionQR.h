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
        explicit DecompositionQR(shared_ptr<Array<double>>& matrix, ParallelizationMethod = Wank, bool storeOnMatrix = false);
        
        void decompose();
        
    protected:
        shared_ptr<Array<double>> _matrix;
        
        shared_ptr<Array<double>> _Q;
        
        shared_ptr<Array<double>> _R;
        
        DecompositionType _decompositionType;
        
        ParallelizationMethod _parallelizationMethod;
        
        bool _storeOnMatrix;

        virtual void _singleThreadDecomposition();
        
        virtual void _multiThreadDecomposition();
        
        virtual void _CUDADecomposition();
    };

} // LinearAlgebra

#endif //UNTITLED_DECOMPOSITIONQR_H
