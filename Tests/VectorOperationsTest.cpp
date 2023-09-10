
#include "../LinearAlgebra/Operations/VectorOperations.h"
#include "../LinearAlgebra/Array/Array.h"
#include "VectorOperationsTest.h"

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace LinearAlgebra;

namespace Tests {
    VectorOperationsTest::VectorOperationsTest(){
        // {1, 2, 3}
        auto vector1 = make_shared<vector<double>>(vector<double>({1, 2, 3}));
        // {1, 2, 3}
        auto vector2 = make_shared<vector<double>>(vector<double>({1, 2, 3}));
        
        auto result = make_shared<vector<double>>(vector<double>({0, 0, 0}));
        // {1, 2, 3, 4, 5, 6, 7, 8, 9}
        auto array = make_shared<Array<double>>(3,3);
        array->at(0,0) = 1;
        array->at(0,1) = 2;
        array->at(0,2) = 3;
        array->at(1,0) = 4;
        array->at(1,1) = 5;
        array->at(1,2) = 6;
        array->at(2,0) = 7;
        array->at(2,1) = 8;
        array->at(2,2) = 9;
        
        

        //14
        auto dotProduct = VectorOperations::dotProduct(vector1, vector2);
        std::cout << "Dot product: " << dotProduct << " Correct Solution : " << 14 << std::endl;

        //{2, 4, 6}
        VectorOperations::add(vector1, vector2, result);
        std::cout << "Addition: " << result->at(0) << ", " << result->at(1) << ", " << result->at(2) <<
        "Correct Solution : " << 2 << ", " << 4 << ", " << 6 << std::endl;
        
        //{0, 0, 0} monit
        VectorOperations::subtract(vector1, vector2, result);
        std::cout << "Subtraction: " << result->at(0) << ", " << result->at(1) << ", " << result->at(2) <<
        "Correct Solution : " << 0 << ", " << 0 << ", " << 0 << std::endl;
        //{14, 32, 50}
        VectorOperations::matrixVectorMultiplication(array, vector1, result);
        std::cout << "NumericalMatrix vector multiplication: " << result->at(0) << ", " << result->at(1) << ", " << result->at(2) <<
        "Correct Solution : " << 14 << ", " << 32 << ", " << 50 << std::endl;
        
        VectorOperations::add(vector1, vector2, result, 2);
        std::cout << "Add scaled vector: " << result->at(0) << ", " << result->at(1) << ", " << result->at(2) <<
        "Correct Solution : " << 3 << ", " << 6 << ", " << 9 << std::endl;
        
        
        VectorOperations::subtract(vector1, vector2, result, 2);
        std::cout << "Subtract scaled vector: " << result->at(0) << ", " << result->at(1) << ", " << result->at(2) <<
        "Correct Solution : " << -1 << ", " << -2 << ", " << -3 << std::endl;
        
        auto sum = VectorOperations::sum(vector1);
        std::cout << "Sum: " << sum << " Correct Solution : " << 6 << std::endl;
        
        auto average = VectorOperations::average(vector1);
        std::cout << "Average: " << average << " Correct Solution : " << 2 << std::endl;
        
        auto magnitude = VectorOperations::magnitude(vector1);
        std::cout << "Magnitude: " << magnitude << " Correct Solution : " << sqrt(14) << std::endl;
        
        VectorOperations::normalize(vector1);
        std::cout << "Normalize: " << vector1->at(0) << ", " << vector1->at(1) << ", " << vector1->at(2) <<
        "Correct Solution : " << 1/sqrt(14) << ", " << 2/sqrt(14) << ", " << 3/sqrt(14) << std::endl;
        
        auto distance = VectorOperations::distance(vector1, vector2);
        std::cout << "Distance: " << distance << " Correct Solution : " << 0 << std::endl;
        
        auto angle = VectorOperations::angle(vector1, vector2);
        std::cout << "Angle: " << angle << " Correct Solution : " << 0 << std::endl;
        
    }
} // Tests

