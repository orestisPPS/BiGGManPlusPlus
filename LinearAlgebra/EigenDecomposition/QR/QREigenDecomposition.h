//
// Created by hal9000 on 8/5/23.
//

#ifndef UNTITLED_QREIGENDECOMPOSITION_H
#define UNTITLED_QREIGENDECOMPOSITION_H

#include "GramSchmidtQR.h"

namespace LinearAlgebra {

    class QREigenDecomposition {
    private:
        QREigenDecomposition(shared_ptr<Array<double>>& matrix, DecompositionType = GramSchmidt,
                             ParallelizationMethod = Wank, bool storeOnMatrix = false);
        
        void calculateEigenValues();
        
    };

} // LinearAlgebra

#endif //UNTITLED_QREIGENDECOMPOSITION_H
