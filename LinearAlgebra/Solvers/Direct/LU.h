//
// Created by hal9000 on 4/24/23.
//

#ifndef UNTITLED_LU_H
#define UNTITLED_LU_H

#include "DirectSolver.h"

namespace LinearAlgebra {

    class LU : public DirectSolver {
        
    public:
        
        explicit LU(bool storeDecompositionOnMatrix);
        
        ~LU();
        
        void solve() override;
        
    private:
        Array<double>* _l;
        
        Array<double>* _u;
    };

} // LinearAlgebra

#endif //UNTITLED_LU_H
