//
// Created by hal9000 on 4/25/23.
//

#include <fstream>
#include <utility>
#include "LinearSystem.h"

namespace LinearAlgebra {
    
    LinearSystem::LinearSystem(shared_ptr<NumericalMatrix<double>> inputMatrix, shared_ptr<NumericalVector<double>> inputRHS) :
            matrix(std::move(inputMatrix)), rhs(std::move(inputRHS)),
            solution(make_shared<NumericalVector<double>>(rhs->size())), logs(Logs("LinearSystem")) {
         _setLogs();
    }
    
    void LinearSystem::exportToMatlabFile(const string& fileName, const string& filePath, bool printSolution) const {

        ofstream outputFile(filePath + fileName);

        // Write the matrix to the file
        outputFile << "A = [";
        for (int i = 0; i < matrix->numberOfRows(); i++) {
            for (int j = 0; j < matrix->numberOfColumns(); j++) {
                outputFile << matrix->getElement(i, j) << " ";
            }
            if (i < matrix->size() - 1) {
                outputFile << "; ";
            }
        }
        outputFile << "];" << endl;

        // Write the vector to the file
        outputFile << "b = [";
        for (double i: *rhs) {
            outputFile << i << " ";
        }
        outputFile << "]';" << endl;

        // Write the command to solve the system
        outputFile << "x = A \\ b" << endl;

        outputFile << "[L, U] = lu(A);" << endl;
        
            outputFile.close();
    }

    void LinearSystem::_setLogs() {
        logs.setSingleObservationLogData("Matrix size", matrix->dataStorage->sizeInKB());
        logs.setSingleObservationLogData("RHS size", rhs->sizeInKB());
        logs.setSingleObservationLogData("Solution size", solution->sizeInKB());
        logs.setSingleObservationLogData("Total size", matrix->dataStorage->sizeInKB() + rhs->sizeInKB() + solution->sizeInKB());
    }
}// LinearAlgebra