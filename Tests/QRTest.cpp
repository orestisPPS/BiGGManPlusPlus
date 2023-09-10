//
// Created by hal9000 on 8/5/23.
//

#include "QRTest.h"
#include "../LinearAlgebra/EigenDecomposition/QR/HouseHolderQR.h"
#include "../LinearAlgebra/EigenDecomposition/QR/IterationQR.h"
using namespace LinearAlgebra;

namespace Tests {
    QRTest::QRTest() {
        auto matrix1 = make_shared<Array<double>>(4, 4);
        matrix1->at(0,0) = 4;
        matrix1->at(0,1) = 1;
        matrix1->at(0,2) = 2;
        matrix1->at(0,3) = 3;

        matrix1->at(1,0) = 1;
        matrix1->at(1,1) = 3;
        matrix1->at(1,2) = 1;
        matrix1->at(1,3) = 2;

        matrix1->at(2,0) = 2;
        matrix1->at(2,1) = 1;
        matrix1->at(2,2) = 6;
        matrix1->at(2,3) = 1;

        matrix1->at(3,0) = 3;
        matrix1->at(3,1) = 2;
        matrix1->at(3,2) = 1;
        matrix1->at(3,3) = 5;
        
        //auto qr = make_shared<LinearAlgebra::GramSchmidtQR>(matrix1);
        auto qr = make_shared<IterationQR>(10, 1E-4, Householder, SingleThread, true);
        qr->setMatrix(matrix1);
        qr->calculateEigenvalues();
        
        auto matrix2 = make_shared<LinearAlgebra::Array<double>>(4, 3);
        matrix2->at(0,0) = 1;
        matrix2->at(0,1) = -1;
        matrix2->at(0,2) = 4;
        
        matrix2->at(1,0) = 1;
        matrix2->at(1,1) = 4;
        matrix2->at(1,2) = -2;
        
        matrix2->at(2,0) = 1;
        matrix2->at(2,1) = 4;
        matrix2->at(2,2) = 2;
        
        matrix2->at(3,0) = 1;
        matrix2->at(3,1) = -1;
        matrix2->at(3,2) = 0;

        //auto qr = make_shared<LinearAlgebra::HouseHolderQR>(matrix2);
        //qr->decompose();

    }
} // Tests