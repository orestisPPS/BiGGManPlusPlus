//
// Created by hal9000 on 9/8/23.
//

#ifndef UNTITLED_EIGENDECOMPOSITIONPROVIDER_H
#define UNTITLED_EIGENDECOMPOSITIONPROVIDER_H

#include "../NumericalMatrix.h"
#include "../../../EigenDecomposition/QR/IterationQR.h"
#include "../../../EigenDecomposition/LanczosEigenDecomposition.h"
#include "../../../EigenDecomposition/PowerMethod.h"


namespace LinearAlgebra {

    class EigendecompositionProvider {
    public:
        EigendecompositionProvider();
        
    private:
/*        unique_ptr<IterationQR> _iterationQR;
        
        unique_ptr<LanczosEigenDecomposition> _lanczosEigenDecomposition;
        
        unique_ptr<PowerMethod> _powerMethod;*/
        
    };

} // LinearAlgebra

#endif //UNTITLED_EIGENDECOMPOSITIONPROVIDER_H
