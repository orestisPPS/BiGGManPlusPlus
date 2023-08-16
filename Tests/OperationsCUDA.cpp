//
// Created by hal9000 on 7/27/23.
//

#include "OperationsCUDA.h"

namespace Tests{

    OperationsCUDA::OperationsCUDA() {
        vector<double>* h_vector1 = new vector<double>({1, 2, 3});
        vector<double>* h_vector2 = new vector<double>({1, 2, 3});
        vector<double>* h_array = new vector<double>({1, 2, 3, 4, 5, 6, 7, 8, 9});
        
        //14
        auto dotProduct = LinearAlgebraCUDA::NumericalOperationsCUDA::dotProduct(h_vector1->data(), h_vector2->data(), 3);
        //{2, 4, 6}
        auto addition = LinearAlgebraCUDA::NumericalOperationsCUDA::vectorAdd(h_vector1->data(), h_vector2->data(), 3);
        //{0, 0, 0} monit
        auto subtraction = LinearAlgebraCUDA::NumericalOperationsCUDA::vectorSubtract(h_vector1->data(), h_vector2->data(), 3);
        //{14, 32, 50}
        auto matrixVector = LinearAlgebraCUDA::NumericalOperationsCUDA::matrixVectorMultiply(h_array->data(), h_vector1->data(), 3, 3);
        
        std::cout << "Dot product: " << dotProduct << std::endl;
        std::cout << "Addition: " << addition[0] << ", " << addition[1] << ", " << addition[2] << std::endl;
        std::cout << "Subtraction: " << subtraction[0] << ", " << subtraction[1] << ", " << subtraction[2] << std::endl;
        std::cout << "Matrix vector multiplication: " << matrixVector[0] << ", " << matrixVector[1] << ", " << matrixVector[2] << std::endl;
        
        
    }

}
