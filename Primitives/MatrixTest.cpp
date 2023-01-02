/*
//
// Created by hal9000 on 11/28/22.
//
#include "Array.h"
#include "MatrixTest.h"
using namespace Primitives;


MatrixTest::MatrixTest() {
    Array<int> matrix(3, 3);
    matrix.populateElement(0, 0, 1);
    matrix.populateElement(0, 1, 2);
    matrix.populateElement(0, 2, 3);
    matrix.populateElement(1, 0, 4);
    std::cout<<"Array: "<<std::endl;
    matrix.print();
    
//    Array<int> transpose = matrix.transpose();
//    std::cout << "Transpose Array" << std::endl;
//    transpose.print();

    //Test = operator
    Array<int> matrix1 = Array<int>(3, 3);
    matrix1 = matrix;
    std::cout << "Matrix1" << std::endl;
    matrix1.print();

    //Test matrix addition
    std::cout << "Array Add" << std::endl;
    auto matrixAdd = matrix + matrix;
    matrixAdd.print();


};*/
