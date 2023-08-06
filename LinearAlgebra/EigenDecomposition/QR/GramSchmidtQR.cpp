//
// Created by hal9000 on 8/5/23.
//

#include "GramSchmidtQR.h"
#include "../../Norms/VectorNorm.h"

namespace LinearAlgebra {
    
    GramSchmidtQR::GramSchmidtQR(shared_ptr<Array<double>> &matrix, ParallelizationMethod parallelizationMethod,
                                 bool storeOnMatrix) : DecompositionQR(matrix, parallelizationMethod, storeOnMatrix){
        _decompositionType = GramSchmidt;
    }

    void GramSchmidtQR::_singleThreadDecomposition() {
        unsigned n = _matrix->numberOfRows();
        //Initialize a vector to store each column
        auto iColumnOfA = make_shared<vector<double>>(n);
        auto jColumnOfQ = make_shared<vector<double>>(n);
        double norm = 0.0;
        double QjDotAi = 0.0;
        
        //March through each column of A to find the projection of A onto Q
        for (unsigned i = 0; i < n; i++){
            //Store column i in iColumn
            iColumnOfA = _matrix->getColumn(i);

            //Calculate the orthogonal projection of column i of A onto the subspace spanned by
            //previously computed columns of Q            
            for (unsigned j = 0; j < i; j++) {
                jColumnOfQ = _Q->getColumn(j);
                QjDotAi = VectorOperations::dotProduct(iColumnOfA, jColumnOfQ);
                _R->at(j, i) = QjDotAi;
                //Subtract the projection of column i of A making it orthogonal to column j of Q
                VectorOperations::scale(jColumnOfQ, QjDotAi);
                VectorOperations::subtractIntoThis(iColumnOfA, jColumnOfQ);
            }
            //Normalize column i of A
            norm = VectorNorm(iColumnOfA, L2).value();
            _R->at(i, i) = norm;
            VectorOperations::normalize(iColumnOfA);
            _Q->setColumn(i, iColumnOfA);
        }
        
        _Q->print(4);
        cout<< endl;
        _R->print(4);
        cout << endl;

    }

    void GramSchmidtQR::_multiThreadDecomposition() {
        DecompositionQR::_multiThreadDecomposition();
    }

    void GramSchmidtQR::_CUDADecomposition() {
        DecompositionQR::_CUDADecomposition();
    }
} // LinearAlgebra