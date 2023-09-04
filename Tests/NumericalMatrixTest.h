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
            NumericalMatrix<double> matrixCSR = NumericalMatrix<double>(5, 5, CSR);
            matrixCSR.setElement(0, 0, 3);
            matrixCSR.setElement(1, 3, 7);
            matrixCSR.setElement(3, 1, 4);
            matrixCSR.setElement(4, 3, 2);
            matrixCSR.setElement(4, 4, 5);

            auto values = matrixCSR.dataStorage->getValues();
            auto columnIndices = matrixCSR.dataStorage->getSupplementaryVectors()[0];
            auto rowOffsets = matrixCSR.dataStorage->getSupplementaryVectors()[1];
            
            vector<double> expectedValues = {3, 7, 4, 2, 5};
            vector<unsigned> expectedRowOffsets = {0, 1, 2, 2, 3, 5};
            vector<unsigned> expectedColumnIndices = {0, 3, 1, 3, 4};
            
            for (unsigned i = 0; i < expectedValues.size(); ++i){
                if (values->at(i) != expectedValues[i])
                    throw runtime_error("Values are not correct.");
                if (rowOffsets->at(i) != expectedRowOffsets[i])
                    throw runtime_error("Row indices are not correct.");
                if (columnIndices->at(i) != expectedColumnIndices[i])
                    throw runtime_error("Column indices are not correct.");
            }
            
            cout << "Test passed" << endl;
        }

        static void testCSRMatrixWithCOOAssignment(){
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

            vector<double> expectedValues = {3, 7, 4, 2, 5};
            vector<unsigned> expectedRowOffsets = {0, 1, 2, 2, 3, 5};
            vector<unsigned> expectedColumnIndices = {0, 3, 1, 3, 4};

            for (unsigned i = 0; i < expectedValues.size(); ++i){
                if (values->at(i) != expectedValues[i])
                    throw runtime_error("Values are not correct.");
                if (rowOffsets->at(i) != expectedRowOffsets[i])
                    throw runtime_error("Row indices are not correct.");
                if (columnIndices->at(i) != expectedColumnIndices[i])
                    throw runtime_error("Column indices are not correct.");
            }

            cout << "Test Passed " << endl;
        }
    };

} // Tests

#endif //UNTITLED_NUMERICALMATRIXTEST_H
