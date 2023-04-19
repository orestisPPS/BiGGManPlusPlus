//
// Created by hal9000 on 4/19/23.
//

#ifndef UNTITLED_FDSCHEMETYPE_H
#define UNTITLED_FDSCHEMETYPE_H

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

} // LinearAlgebra

#endif //UNTITLED_FDSCHEMETYPE_H
