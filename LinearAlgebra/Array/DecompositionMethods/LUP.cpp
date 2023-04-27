//
// Created by hal9000 on 4/26/23.
//

#include "LUP.h"

namespace LinearAlgebra {

    LUP::LUP(Array<double>* matrix, double pivotTolerance, bool throwExceptionOnSingularMatrix) :
            _matrix(matrix),
            _pivotTolerance(pivotTolerance),
            _throwExceptionOnSingularMatrix(throwExceptionOnSingularMatrix),
            _l(nullptr),
            _u(nullptr),
            _p(nullptr),
            _isSingular(false) {
    }

    bool LUP::_runMatrixDiagnostics() {
        if (_matrix->isSquare()) {
            return true;
        } else {
            cout << "Matrix is not square." << endl;
            return false;
        }
        //Check for 
    }

    LUP::~LUP() {
        if (_l != nullptr) {
            delete _l;
        }
        if (_u != nullptr) {
            delete _u;
        }
        if (_p != nullptr) {
            delete _p;
        }
    }

    void LUP::decompose(bool deleteMatrixAfterDecomposition) {
        unsigned n = _matrix->numberOfRows();
        _l = new Array<double>(n, n);
        _u = new Array<double>(n, n);
        _p = new vector<unsigned>(n, 1);

        unsigned i, j, k, iMax;
        double maxA, absA;


        // Loop over each column in the matrix
        for (i = 0; i < n; ++i) {
            // Find the row with the maximum absolute value for the current column
            maxA = 0.0;
            iMax = i;
            for (k = i; k < n; k++) {
                absA = fabs(_matrix->at(k, i));
                if (absA > maxA) {
                    maxA = absA;
                    iMax = k;
                }
            }
            // Check if the matrix is singular
            _isSingular = maxA < _pivotTolerance;
            if (_isSingular && _throwExceptionOnSingularMatrix) {
                // Throw an exception if the matrix is singular and the flag is set
                throw runtime_error("WARNING: Matrix is singular. It is degenerate like yourself.");
            }
            else if (_isSingular && !_throwExceptionOnSingularMatrix) {
                // Print a warning message if the matrix is singular and the flag is not set
                cout << "WARNING: Matrix is singular. It is degenerate like yourself." << endl;
                break;
            }

            if (iMax != i) {
                // Swap the rows of P, A, and L if the row with the maximum absolute value is not the current row
                j = _p->at(i);
                _p->at(i) = _p->at(iMax);
                _p->at(iMax) = j;
                _matrix->swapRows(i, iMax);
                _l->swapRows(i, iMax);
            }
            // Calculate the elements of L and U matrices
            for (j = i + 1; j < n; j++) {
                _l->at(j, i) = _matrix->at(j, i) / _matrix->at(i, i);
                for (k = i + 1; k < n; k++) {
                    _matrix->at(j, k) -= _l->at(j, i) * _matrix->at(i, k);
                }
            }
        }
        if (!_isSingular){
            // Fill diagonal of L with 1's
            for (i = 0; i < n; i++) {
                _l->at(i, i) = 1.0;
            }

            // Copy upper triangular elements to U
            for (i = 0; i < n; i++) {
                for (j = i; j < n; j++) {
                    _u->at(i, j) = _matrix->at(i, j);
                }
            }

            // Delete the original matrix if specified
            if (deleteMatrixAfterDecomposition) {
                delete _matrix;
            }
        }
    }

    void LUP::decomposeOnMatrix() {
        unsigned n = _matrix->numberOfRows();
        // Initialize the permutation vector P to the identity.
        _p = new vector<unsigned>(n, 1);

        // Declare some loop variables
        unsigned i, j, k, iMax;
        double maxA, absA;

        for (i = 0; i < n; i++) {
            _p->at(i) = i;
        }

        // Loop over each column of the matrix
        for (i = 0; i < n ; ++i) {
            // Find the row with the largest absolute value in the current column
            maxA = 0.0;
            iMax = i;
            for(k = i; k < n; k++) {
                absA = fabs(_matrix->at(k, i));
                if (absA > maxA) {
                    maxA = absA;
                    iMax = k;
                }
            }

            // Check if the matrix is singular
            _isSingular = maxA < _pivotTolerance;
            if (_isSingular && _throwExceptionOnSingularMatrix) {
                throw runtime_error("WARNING: Matrix is singular. It is degenerate like yourself.");
            }
            else if (_isSingular && !_throwExceptionOnSingularMatrix) {
                cout << "WARNING: Matrix is singular. It is degenerate like yourself." << endl;
                break;
            }

            // Swap the rows of the matrix, L, and P if necessary
            if (iMax != i) {
                //Pivoting P
                j = _p->at(i);
                _p->at(i) = _p->at(iMax);
                _p->at(iMax) = j;

                //Pivoting A rows
                _matrix->swapRows(i, iMax);

                // Increment the last element of the permutation vector
                _p->at(n)++;
            }

            // Update the lower triangular matrix L and the upper triangular matrix U
            for (j = i + 1; j < n; j++){
                _matrix->at(j, i) /= _matrix->at(i, i);
                for (k = i + 1; k < n; k++) {
                    _matrix->at(j, k) -= _matrix->at(j, i) * _matrix->at(i, k);
                }
            }
        }
    }

    unique_ptr<Array<double>> LUP::getL() {
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

    unique_ptr<Array<double>> LUP::getU() {
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

    unique_ptr<vector<unsigned >> LUP::getP() {
        return unique_ptr<vector<unsigned >>(_p);
    }

    Array<double>* LUP::invertMatrix() {
        unsigned n = _matrix->numberOfRows();
        auto inverse = new Array<double>(n, n);

        for(unsigned j = 0; j < n; j++){
            for (unsigned i = 0; i < n; i++){
                inverse->at(i, j) = _p->at(i) == j ? 1.0 : 0.0;
                for (unsigned k = 0; k < i; k++){
                    inverse->at(i, j) -= _l->at(i, k) * inverse->at(k, j);
                }
            }
            for (int i = static_cast<int>(n) - 1; i >= 0; i--){
                for (unsigned k = i + 1; k < n; k++){
                    inverse->at(i, j) -= _u->at(i, k) * inverse->at(k, j);
                }
                inverse->at(i, j) /= _u->at(i, i);
            }
        }
        return inverse;
    }

    double LUP::determinant() {
        auto det = _matrix->at(0, 0);
        if (isStoredOnMatrix()){
            for (unsigned i = 1; i < _matrix->numberOfRows(); i++) {
                det *= _matrix->at(i, i);
            }
        }
        else {
            for (unsigned i = 1; i < _l->numberOfRows(); i++) {
                det *= _l->at(i, i);
            }
        }
        auto n = _matrix->numberOfRows() - 1;
        return (_p->at(n) - n) % 2 == 0 ? det : -det;
    }

    vector<double>* LUP::solve(vector<double>* rhs) {
        unsigned n = _matrix->numberOfRows();
        auto x = new vector<double>(n);
        auto y = new vector<double>(n);

        // Solve Ly = b using forward substitution
        for (unsigned i = 0; i < n; i++) {
            y->at(i) = (*rhs)[_p->at(i)];
            for (unsigned j = 0; j < i; j++) {
                y->at(i) -= _l->at(i, j) * (*y)[j];
            }
        }

        // Solve Ux = y using backward substitution
        for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
            x->at(i) = y->at(i);
            for (unsigned j = i + 1; j < n; j++) {
                x->at(i) -= _u->at(i, j) * (*x)[j];
            }
            x->at(i) /= _u->at(i, i);
        }
        delete y;
        return x;
    }

    bool LUP::isStoredOnMatrix(){
        if (_l == nullptr && _u == nullptr) {
            return false;
        }
        else {
            return true;
        }
    }

} // LinearAlgebra