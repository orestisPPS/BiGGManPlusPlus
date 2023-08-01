//
// Created by hal9000 on 4/27/23.
//

#include "MatrixDecomposition.h"

#include <utility>

namespace LinearAlgebra {

    MatrixDecomposition::MatrixDecomposition(shared_ptr<Array<double>> matrix) {
        _matrix = std::move(matrix);
        _l = nullptr;
        _u = nullptr;
    }
    

    shared_ptr<Array<double>> MatrixDecomposition::getL() {
        if (_l == nullptr) {
            // create _l
            unsigned n = _matrix->numberOfRows();
            _l = make_shared<Array<double>>(n, n);
            for (unsigned i = 0; i < n; i++) {
                _l->at(i, i) = 1.0;
                for (unsigned j = 0; j <= i; j++) {
                    _l->at(i, j) = _matrix->at(i, j);
                }
            }
        }
        return _l;
    }
    shared_ptr<Array<double>> MatrixDecomposition::getU() {
        if (_u == nullptr) {
            // create _u
            unsigned n = _matrix->numberOfRows();
            _u = make_shared<Array<double>>(n, n);
            for (unsigned i = 0; i < n; i++) {
                for (unsigned j = i; j < n; j++) {
                    _u->at(i, j) = _matrix->at(i, j);
                }
            }
        }
        return _u;
    }

    bool MatrixDecomposition::isStoredOnMatrix(){
        if (_l == nullptr && _u == nullptr) {
            return true;
        }
        else {
            return false;
        }
    }
    
    void MatrixDecomposition::decompose() {
        
    }

    void MatrixDecomposition::decomposeOnMatrix() {
        
    }
    
    shared_ptr<Array<double>> MatrixDecomposition::invertMatrix() {
        return nullptr;
    }
    
    double MatrixDecomposition::determinant() {
        return 0;
    }
    
    shared_ptr<vector<double>> MatrixDecomposition::solve(shared_ptr<vector<double>> rhs, shared_ptr<vector<double>> solution) {
    }

} // LinearAlgebra