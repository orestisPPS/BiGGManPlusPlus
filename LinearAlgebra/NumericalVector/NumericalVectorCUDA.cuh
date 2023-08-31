//
// Created by hal9000 on 8/31/23.
//

#ifndef UNTITLED_NUMERICALVECTORCUDA_CUH
#define UNTITLED_NUMERICALVECTORCUDA_CUH

#include <memory>


namespace LinearAlgebra {

    class NumericalVectorCUDA {
    public:
        NumericalVectorCUDA(unsigned int size, T initialValue = 0, );
        ~NumericalVectorCUDA();
    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALVECTORCUDA_CUH
