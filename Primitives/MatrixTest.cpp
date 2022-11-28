//
// Created by hal9000 on 11/28/22.
//
#include "Matrix.h"
#include "MatrixTest.h"
using namespace Primitives;


MatrixTest::MatrixTest() {
    Matrix<int> matrix(3, 3);
    matrix.populateElement(0, 0, 1);
    matrix.populateElement(0, 1, 2);
    matrix.populateElement(0, 2, 3);
    matrix.populateElement(1, 0, 4);
    std::cout<<"Matrix: "<<std::endl;
    matrix.print();
    
//    Matrix<int> transpose = matrix.transpose();
//    std::cout << "Transpose Matrix" << std::endl;
//    transpose.print();

    //Test = operator
    Matrix<int> matrix1 = Matrix<int>(3, 3);
    matrix1 = matrix;
    std::cout << "Matrix1" << std::endl;
    matrix1.print();

    //Test matrix addition
    std::cout << "Matrix Add" << std::endl;
    auto matrixAdd = matrix + matrix;
    matrixAdd.print();


};