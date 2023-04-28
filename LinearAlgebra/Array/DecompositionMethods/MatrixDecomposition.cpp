//
// Created by hal9000 on 4/27/23.
//

#include "MatrixDecomposition.h"

namespace LinearAlgebra {

    MatrixDecomposition::MatrixDecomposition(Array<double>* matrix) {
        _matrix = matrix;
        _l = nullptr;
        _u = nullptr;
    }
    
    MatrixDecomposition::~MatrixDecomposition() {
        if (_l != nullptr) {
            delete _l;
        }
        if (_u != nullptr) {
            delete _u;
        }
    }
    
    unique_ptr<Array<double>> MatrixDecomposition::getL() {
        if (_l == nullptr) {
            // create _l
            unsigned n = _matrix->numberOfRows();
            _l = new Array<double>(n, n);
            for (unsigned i = 0; i < n; i++) {
                _l->at(i, i) = 1.0;
                for (unsigned j = 0; j <= i; j++) {
                    _l->at(i, j) = _matrix->at(i, j);
                }
            }
        }
        return unique_ptr<Array<double>>(_l);
    }

    unique_ptr<Array<double>> MatrixDecomposition::getU() {
        if (_u == nullptr) {
            // create _u
            unsigned n = _matrix->numberOfRows();
            _u = new Array<double>(n, n);
            for (unsigned i = 0; i < n; i++) {
                for (unsigned j = i; j < n; j++) {
                    _u->at(i, j) = _matrix->at(i, j);
                }
            }
        }
        return unique_ptr<Array<double>>(_u);
    }

    bool MatrixDecomposition::isStoredOnMatrix(){
        if (_l == nullptr && _u == nullptr) {
            return true;
        }
        else {
            return false;
        }
    }
    
    void MatrixDecomposition::decompose(bool deleteMatrixAfterDecomposition) {
        
    }

    void MatrixDecomposition::decomposeOnMatrix() {
        
    }
    
    Array<double>* MatrixDecomposition::invertMatrix() {
        return nullptr;
    }
    
    double MatrixDecomposition::determinant() {
        return 0;
    }
    
    vector<double>* MatrixDecomposition::solve(vector<double>* b) {
        return nullptr;
    }

} // LinearAlgebra