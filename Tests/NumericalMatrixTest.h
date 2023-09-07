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
            testFullMatrixElementAssignment();
            testCSRMatrixWithOnSpotElementAssignment();
            testCSRMatrixWithCOOElementAssignment();
            testMatrixAddition();
            testMatrixSubtraction();
            testMatrixMultiplication();
            testMatrixVectorMultiplication();
            testMatrixAdditionMultiThread();
            testMatrixSubtractionMultiThread();
            testMatrixMultiplicationMultiThread();
            testMatrixVectorMultiplicationMultiThread();
            
        }
        
        static void testFullMatrixElementAssignment(){
            logTestStart("testFullMatrixElementAssignment");
            NumericalMatrix<double> matrixFull = NumericalMatrix<double>(5, 5, FullMatrix);
            matrixFull.setElement(0, 0, 3);
            matrixFull.setElement(1, 3, 7);
            matrixFull.setElement(3, 1, 4);
            matrixFull.setElement(4, 3, 2);
            matrixFull.setElement(4, 4, 5);

            auto values = matrixFull.dataStorage->getValues();

            NumericalVector<double> expectedValues = {3, 0, 0, 0, 0,
                                                      0, 0, 0, 7, 0,
                                                      0, 0, 0, 0, 0,
                                                      0, 4, 0, 0, 0,
                                                      0, 0, 0, 2, 5};

            assert(expectedValues == values);
            
            logTestEnd();
        }

        static void testMatrixAddition() {
            logTestStart("testMatrixAddition");

            NumericalMatrix<double> matrixA(2, 2, FullMatrix);
            matrixA.setElement(0, 0, 1);
            matrixA.setElement(0, 1, 2);
            matrixA.setElement(1, 0, 3);
            matrixA.setElement(1, 1, 4);

            NumericalMatrix<double> matrixB(2, 2, FullMatrix);
            matrixB.setElement(0, 0, 4);
            matrixB.setElement(0, 1, 3);
            matrixB.setElement(1, 0, 2);
            matrixB.setElement(1, 1, 1);

            NumericalMatrix<double> resultMatrix = NumericalMatrix<double>(2, 2, FullMatrix);
            matrixA.add(matrixB, resultMatrix);

            NumericalVector<double> expectedValues = {5, 5, 5, 5};
            auto values = resultMatrix.dataStorage->getValues();
            assert(expectedValues == values);

            logTestEnd();
        }

        static void testMatrixSubtraction() {
            logTestStart("testMatrixSubtraction");

            NumericalMatrix<double> matrixA(2, 2, FullMatrix);
            matrixA.setElement(0, 0, 5);
            matrixA.setElement(0, 1, 4);
            matrixA.setElement(1, 0, 3);
            matrixA.setElement(1, 1, 2);

            NumericalMatrix<double> matrixB(2, 2, FullMatrix);
            matrixB.setElement(0, 0, 1);
            matrixB.setElement(0, 1, 2);
            matrixB.setElement(1, 0, 3);
            matrixB.setElement(1, 1, 4);

            NumericalMatrix<double> resultMatrix = NumericalMatrix<double>(2, 2, FullMatrix);
            matrixA.subtract(matrixB, resultMatrix);

            NumericalVector<double> expectedValues = {4, 2, 0, -2};
            auto values = resultMatrix.dataStorage->getValues();
            assert(expectedValues == values);

            logTestEnd();
        }

        static void testMatrixMultiplication() {
            logTestStart("testMatrixMultiplication");

            NumericalMatrix<double> matrixA(2, 2, FullMatrix);
            matrixA.setElement(0, 0, 1);
            matrixA.setElement(0, 1, 2);
            matrixA.setElement(1, 0, 3);
            matrixA.setElement(1, 1, 4);

            NumericalMatrix<double> matrixB(2, 2, FullMatrix);
            matrixB.setElement(0, 0, 2);
            matrixB.setElement(0, 1, 0);
            matrixB.setElement(1, 0, 1);
            matrixB.setElement(1, 1, 3);

            NumericalMatrix<double> resultMatrix = NumericalMatrix<double>(2, 2, FullMatrix);
            matrixA.multiplyMatrix(matrixB, resultMatrix);

            NumericalVector<double> expectedValues = {4, 6, 10, 12};
            auto values = resultMatrix.dataStorage->getValues();
            assert(expectedValues == values);

            logTestEnd();
        }

        static void testMatrixVectorMultiplication() {
            logTestStart("testMatrixVectorMultiplication");

            NumericalMatrix<double> matrix(2, 2, FullMatrix);
            matrix.setElement(0, 0, 1);
            matrix.setElement(0, 1, 2);
            matrix.setElement(1, 0, 3);
            matrix.setElement(1, 1, 4);

            NumericalVector<double> vector = {2, 3};

            NumericalVector<double> resultVector = NumericalVector<double>(2);
            
            matrix.multiplyVector(vector, resultVector);

            NumericalVector<double> expectedValues = {8, 18};
            assert(expectedValues == resultVector);

            logTestEnd();
        }

        static void testMatrixAdditionMultiThread() {
            logTestStart("testMatrixAdditionMultiThread");

            auto fullGasBaby = std::thread::hardware_concurrency();
            
            NumericalMatrix<double> matrixA(2, 2, FullMatrix, fullGasBaby);
            matrixA.setElement(0, 0, 1);
            matrixA.setElement(0, 1, 2);
            matrixA.setElement(1, 0, 3);
            matrixA.setElement(1, 1, 4);

            NumericalMatrix<double> matrixB(2, 2, FullMatrix, fullGasBaby);
            matrixB.setElement(0, 0, 4);
            matrixB.setElement(0, 1, 3);
            matrixB.setElement(1, 0, 2);
            matrixB.setElement(1, 1, 1);

            NumericalMatrix<double> resultMatrix = NumericalMatrix<double>(2, 2, FullMatrix, fullGasBaby);
            matrixA.add(matrixB, resultMatrix);

            NumericalVector<double> expectedValues = {5, 5, 5, 5};
            auto values = resultMatrix.dataStorage->getValues();
            assert(expectedValues == values);
            assert(resultMatrix.dataStorage->getAvailableThreads() == fullGasBaby);

            logTestEnd();
        }

        static void testMatrixSubtractionMultiThread() {
            logTestStart("testMatrixSubtractionMultiThread");

            auto fullGasBaby = std::thread::hardware_concurrency();

            NumericalMatrix<double> matrixA(2, 2, FullMatrix, fullGasBaby);
            matrixA.setElement(0, 0, 5);
            matrixA.setElement(0, 1, 4);
            matrixA.setElement(1, 0, 3);
            matrixA.setElement(1, 1, 2);

            NumericalMatrix<double> matrixB(2, 2, FullMatrix, fullGasBaby);
            matrixB.setElement(0, 0, 1);
            matrixB.setElement(0, 1, 2);
            matrixB.setElement(1, 0, 3);
            matrixB.setElement(1, 1, 4);

            NumericalMatrix<double> resultMatrix = NumericalMatrix<double>(2, 2, FullMatrix, fullGasBaby);
            matrixA.subtract(matrixB, resultMatrix);

            NumericalVector<double> expectedValues = {4, 2, 0, -2};
            auto values = resultMatrix.dataStorage->getValues();
            assert(expectedValues == values);
            assert(resultMatrix.dataStorage->getAvailableThreads() == fullGasBaby);

            logTestEnd();
        }

        static void testMatrixMultiplicationMultiThread() {
            logTestStart("testMatrixMultiplicationMultiThread");
            
            auto fullGasBaby = std::thread::hardware_concurrency();
            
            NumericalMatrix<double> matrixA(2, 2, FullMatrix, fullGasBaby);
            matrixA.setElement(0, 0, 1);
            matrixA.setElement(0, 1, 2);
            matrixA.setElement(1, 0, 3);
            matrixA.setElement(1, 1, 4);
            

            NumericalMatrix<double> matrixB(2, 2, FullMatrix, fullGasBaby);
            matrixB.setElement(0, 0, 2);
            matrixB.setElement(0, 1, 0);
            matrixB.setElement(1, 0, 1);
            matrixB.setElement(1, 1, 3);

            NumericalMatrix<double> resultMatrix = NumericalMatrix<double>(2, 2, FullMatrix, fullGasBaby);
            matrixA.multiplyMatrix(matrixB, resultMatrix);

            NumericalVector<double> expectedValues = {4, 6, 10, 12};
            auto values = resultMatrix.dataStorage->getValues();
            assert(expectedValues == values);
            assert(resultMatrix.dataStorage->getAvailableThreads() == fullGasBaby);

            logTestEnd();
        }

        static void testMatrixVectorMultiplicationMultiThread() {
            logTestStart("testMatrixVectorMultiplicationMultiThread");

            auto fullGasBaby = std::thread::hardware_concurrency(); 
            
            NumericalMatrix<double> matrix(2, 2, FullMatrix, fullGasBaby);
            matrix.setElement(0, 0, 1);
            matrix.setElement(0, 1, 2);
            matrix.setElement(1, 0, 3);
            matrix.setElement(1, 1, 4);

            NumericalVector<double> vector = {2, 3};

            NumericalVector<double> resultVector = NumericalVector<double>(2);

            matrix.multiplyVector(vector, resultVector);

            NumericalVector<double> expectedValues = {8, 18};
            assert(expectedValues == resultVector);
            assert(matrix.dataStorage->getAvailableThreads() == fullGasBaby);
            logTestEnd();
        }


        static void testCSRMatrixWithOnSpotElementAssignment(){
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

        static void testCSRMatrixWithCOOElementAssignment(){
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
