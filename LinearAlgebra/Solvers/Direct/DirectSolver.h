//
// Created by hal9000 on 4/25/23.
//

#ifndef UNTITLED_DIRECTSOLVER_H
#define UNTITLED_DIRECTSOLVER_H

#include "../Solver.h"

namespace LinearAlgebra {

    class DirectSolver : public Solver {
        
    public:
        
        explicit DirectSolver(bool storeDecompositionOnMatrix);
        
    protected:
        
        bool _storeDecompositionOnMatrix;

        /**
         * @brief _forwardSubstitution
         * @param L (lower triangular matrix)
         * @param RHS (right hand side vector of the linear system)
         * @return solution vector (y) -> L*y = RHS
        */
        static vector<double>* _forwardSubstitution(Array<double>* L, const vector<double>* RHS);
        
        /**
         * @brief _backwardSubstitution
         * @param U (upper triangular matrix)
         * @param RHS (right hand side vector of the linear system)
         * @return b (solution vector) -> U*b = RHS
        */
        static vector<double>* _backwardSubstitution(Array<double>* U, const vector<double>* RHS);

    };

} // LinearAlgebra

#endif //UNTITLED_DIRECTSOLVER_H
