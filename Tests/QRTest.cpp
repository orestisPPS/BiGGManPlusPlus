//
// Created by hal9000 on 8/5/23.
//

#include "QRTest.h"

namespace Tests {
    QRTest::QRTest() {
        auto matrix = make_shared<LinearAlgebra::Array<double>>(4, 4);
/*        matrix->at(0,0) = 2;
        matrix->at(1,1) = 2;
        matrix->at(2,2) = 2;
        matrix->at(3,3) = 2;*/
        matrix->at(0,0) = 4;
        matrix->at(0,1) = 1;
        matrix->at(0,2) = 2;
        matrix->at(0,3) = 3;

        matrix->at(1,0) = 1;
        matrix->at(1,1) = 3;
        matrix->at(1,2) = 1;
        matrix->at(1,3) = 2;

        matrix->at(2,0) = 2;
        matrix->at(2,1) = 1;
        matrix->at(2,2) = 6;
        matrix->at(2,3) = 1;

        matrix->at(3,0) = 3;
        matrix->at(3,1) = 2;
        matrix->at(3,2) = 1;
        matrix->at(3,3) = 5;
        
        auto qr = make_shared<LinearAlgebra::GramSchmidtQR>(matrix);
        qr->decompose();

    }
} // Tests