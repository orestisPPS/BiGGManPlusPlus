//
// Created by hal9000 on 8/6/23.
//

#include "HouseHolderQR.h"

namespace LinearAlgebra {
    HouseHolderQR::HouseHolderQR(shared_ptr<Array<double>> &matrix, ParallelizationMethod parallelizationMethod,
                                 bool storeOnMatrix) : DecompositionQR(matrix, parallelizationMethod, storeOnMatrix){
        _decompositionType = GramSchmidt;
        if (!_storeOnMatrix){
            unsigned n = _matrix->numberOfRows();
            unsigned m = _matrix->numberOfColumns();
            _Q = make_shared<Array<double>>(n,n);
            _R = make_shared<Array<double>>(n,m);
        }
    }

    void HouseHolderQR::_singleThreadDecomposition() {
        unsigned n = _matrix->numberOfRows();
        unsigned m = _matrix->numberOfColumns();
        
        //R = A TODO : fix this with copy constructor
        double* dataA = _matrix->getArrayPointer();
        double* dataR = _R->getArrayPointer();
        for (unsigned i = 0; i < _matrix->size(); i++){
            dataR[i] = dataA[i];
        }
        
        //Q = I
        for (unsigned i = 0; i < n; i++){
            _Q->at(i,i) = 1.0;
        }
        
        double norm = 0.0;
        auto alpha = 0.0;
        
/*        _Q->print();*/
/*        _R->print();*/
        
        for (unsigned i = 0; i < m - 1; i++){
            //Extract the i-th column of R starting at row i to n-1;
            auto iColumnOfR = _R->getColumnPartial(i, i, m);
            
            //Calculate the norm of the i-th column of R
            norm = VectorNorm(iColumnOfR, L2).value();
            
            //α = -sign(R[i,i]) * norm = -sign(iColumnOfR[0]) * norm
            alpha = -sign(iColumnOfR->at(0)) * norm;
            
            //Calculate the Householder vector v
            auto v = make_shared<vector<double>>(iColumnOfR->size());
            v->at(0) = iColumnOfR->at(0) + alpha;
            for (unsigned j = 1; j < iColumnOfR->size(); j++){
                v->at(j) = iColumnOfR->at(j);
            }
            VectorOperations::normalize(v);
            
/*            auto subMatrixR = _R->getSubMatrixPtr(i, m - 1, i, m - 1);*/
/*            auto subMatrixQ = _Q->getSubMatrixPtr(i, m - 1, i, m - 1);*/

            auto H = VectorOperations::tensorProduct(v, v);
            H->scale(-2.0);
            for (unsigned j = 0; j < H->numberOfRows(); j++){
                H->at(j,j) += 1.0;
            }
            H->print();

           // // Apply Householder transformation H to subMatrixR
           // for (unsigned j = 0; j < subMatrixR->numberOfRows(); j++){
           //     for (unsigned k = 0; k < subMatrixR->numberOfColumns(); k++){
           //         double sum = 0.0;
           //         for (unsigned l = 0; l < H->numberOfColumns(); l++){
           //             sum += H->at(j, l) * subMatrixR->at(l, k);
           //         }
           //         subMatrixR->at(j, k) = sum;
           //     }
           // }
           //
           // // Apply Householder transformation H to subMatrixQ
           // for (unsigned j = 0; j < subMatrixQ->numberOfRows(); j++){
           //     for (unsigned k = 0; k < subMatrixQ->numberOfColumns(); k++){
           //         double sum = 0.0;
           //         for (unsigned l = 0; l < H->numberOfColumns(); l++){
           //             sum += H->at(j, l) * subMatrixQ->at(l, k);
           //         }                          
           //         subMatrixQ->at(j, k) = sum;
           //     }
           // }
            
/*            subMatrixQ->print();
            cout<<endl;
            subMatrixR->print();
            cout<<endl;*/

            

/*            // Directly apply Householder reflection to R without explicitly forming H
            for (unsigned j = 0; j < subMatrixR->numberOfRows(); j++){
                for (unsigned k = 0; k < subMatrixR->numberOfColumns(); k++){
                    double proj = 0.0;
                    for (unsigned l = 0; l < v->size(); l++){
                        proj += v->at(l) * subMatrixR->at(l,k);
                    }
                    subMatrixR->at(j,k) -= 2.0 * v->at(j) * proj;
                }
            }

            // Now update Q with the Householder reflection
            for (unsigned j = 0; j < subMatrixQ->numberOfRows(); j++){
                for (unsigned k = 0; k < subMatrixQ->numberOfColumns(); k++){
                    double proj = 0.0;
                    for (unsigned l = 0; l < v->size(); l++){
                        proj += v->at(l) * subMatrixQ->at(l,k);
                    }
                    subMatrixQ->at(j,k) -= 2.0 * v->at(j) * proj;
                }
            }*/
/*            cout<<"before"<<endl;
            _Q->print();
            cout<<endl;
            _R->print();
            cout<<endl;*/
//            _R->setSubMatrix(i, n - 1, i, n - 1, subMatrixR);
//            _Q->setSubMatrix(i, n - 1, i, n - 1, subMatrixQ);
            
/*            cout<<"after"<<endl;
            _Q->print();
            cout<<endl;
            _R->print();
            cout<<endl;*/
        }

        _Q->print(4);
        cout<< endl;
        _R->print(4);
        cout << endl;

    }

    void HouseHolderQR::_multiThreadDecomposition() {
    }

    void HouseHolderQR::_CUDADecomposition() {
    } 
    
    int HouseHolderQR::sign(double x){
        if (x >= 0.0){
            return 1;
        }
        else{
            return -1;
        }
    }
    
} // LinearAlgebra