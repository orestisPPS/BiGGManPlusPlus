//
// Created by hal9000 on 1/28/23.
//

#ifndef UNTITLED_FDSCHEME_H
#define UNTITLED_FDSCHEME_H

namespace LinearAlgebra {

    //Finite Difference Scheme Type
    //The categorization is based on the location of the points used for approximation with respect to the point at
    // which the derivative is being calculated.
    enum FDSchemeType{
        Forward,
        Backward,
        Central,
        Mixed
    };

    class FDScheme {

    };

} // LinearAlgebra

#endif //UNTITLED_FDSCHEME_H
