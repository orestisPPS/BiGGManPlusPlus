//
// Created by hal9000 on 9/4/23.
//

#ifndef UNTITLED_NUMERICALMATRIXTEST_H
#define UNTITLED_NUMERICALMATRIXTEST_H
#include "../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
namespace Tests {

    class NumericalMatrixTest {
    public:
        NumericalMatrixTest(){
            runTests();
        }
        static void runTests(){
            testCSRMatrixWithOnSpotAssignment();
            testCSRMatrixWithCOOAssignment();
        }
        
        static void testCSRMatrixWithOnSpotAssignment(){
            logTestStart("testCSRMatrixWithOnSpotAssignment");
            NumericalMatrix<double> matrixCSR = NumericalMatrix<double>(5, 5, CSR);
            matrixCSR.setElement(0, 0, 3);
            matrixCSR.setElement(1, 3, 7);
            matrixCSR.setElement(3, 1, 4);
            matrixCSR.setElement(4, 3, 2);
            matrixCSR.setElement(4, 4, 5);

            auto values = matrixCSR.dataStorage->getValues();
            auto columnIndices = matrixCSR.dataStorage->getSupplementaryVectors()[0];
            auto rowOffsets = matrixCSR.dataStorage->getSupplementaryVectors()[1];

            NumericalVector<double> expectedValues = {3, 7, 4, 2, 5};
            NumericalVector<unsigned> expectedRowOffsets = {0, 1, 2, 2, 3, 5};
            NumericalVector<unsigned> expectedColumnIndices = {0, 3, 1, 3, 4};

            assert(expectedValues == values);
            assert(expectedRowOffsets == rowOffsets);
            assert(expectedColumnIndices == columnIndices);
            
            logTestEnd();
        }

        static void testCSRMatrixWithCOOAssignment(){
            logTestStart("testCSRMatrixWithCOOAssignment");
            NumericalMatrix<double> matrixCSR = NumericalMatrix<double>(5, 5, CSR);
            matrixCSR.dataStorage->initializeElementAssignment();
            matrixCSR.setElement(0, 0, 3);
            matrixCSR.setElement(1, 3, 7);
            matrixCSR.setElement(3, 1, 4);
            matrixCSR.setElement(4, 3, 2);
            matrixCSR.setElement(4, 4, 5);
            matrixCSR.dataStorage->finalizeElementAssignment();

            auto values = matrixCSR.dataStorage->getValues();
            auto columnIndices = matrixCSR.dataStorage->getSupplementaryVectors()[0];
            auto rowOffsets = matrixCSR.dataStorage->getSupplementaryVectors()[1];

            NumericalVector<double> expectedValues = {3, 7, 4, 2, 5};
            NumericalVector<unsigned> expectedRowOffsets = {0, 1, 2, 2, 3, 5};
            NumericalVector<unsigned> expectedColumnIndices = {0, 3, 1, 3, 4};

            assert(expectedValues == values);
            assert(expectedRowOffsets == rowOffsets);
            assert(expectedColumnIndices == columnIndices);
            
            logTestEnd();
        }

        static void logTestStart(const std::string& testName) {
            std::cout << "Running " << testName << "... ";
        }

        static void logTestEnd() {
            std::cout << "\033[1;32m[PASSED]\033[0m\n";  // This adds a green [PASSED] indicator
        }
    };

} // Tests

#endif //UNTITLED_NUMERICALMATRIXTEST_H
