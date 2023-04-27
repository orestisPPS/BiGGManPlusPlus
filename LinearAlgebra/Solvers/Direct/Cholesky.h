//
// Created by hal9000 on 4/25/23.
//

#ifndef UNTITLED_CHOLESKY_H
#define UNTITLED_CHOLESKY_H

#include "DirectSolver.h"

namespace LinearAlgebra {

    class Cholesky : public DirectSolver {

    public:

        explicit Cholesky(bool storeDecompositionOnMatrix);

        ~Cholesky();

        void solve() override;

    private:
        Array<double>* _l;

        Array<double>* _lT;
    };

} // LinearAlgebra

#endif //UNTITLED_CHOLESKY_H
