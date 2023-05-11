//
// Created by hal9000 on 4/25/23.
//

#include <fstream>
#include "LinearSystem.h"

namespace LinearAlgebra {
    
    LinearSystem::LinearSystem(Array<double> *inputMatrix, vector<double> *inputRHS) :
            matrix(inputMatrix),
            RHS(inputRHS),
            solution(nullptr) {}
    
    LinearSystem::~LinearSystem() {
        delete matrix;
        delete RHS;
        delete solution;
    }


    void LinearSystem::exportToMatlabFile(const string& fileName, const string& filePath, bool printSolution) const {

        ofstream outputFile(filePath + fileName);

        // Write the matrix to the file
        outputFile << "A = [";
        for (int i = 0; i < matrix->numberOfRows(); i++) {
            for (int j = 0; j < matrix->numberOfColumns(); j++) {
                outputFile << matrix->at(i, j) << " ";
            }
            if (i < matrix->size() - 1) {
                outputFile << "; ";
            }
        }
        outputFile << "];" << endl;

        // Write the vector to the file
        outputFile << "b = [";
        for (double i: *RHS) {
            outputFile << i << " ";
        }
        outputFile << "]';" << endl;

        // Write the command to solve the system
        outputFile << "x = A \\ b;" << endl;

        // Write the command to solve the system
        outputFile << "x = A \\ b;" << endl;

        outputFile << "[L, U] = lu(A);" << endl;


        if (printSolution) {
            // Write the command to create a grid for evaluating the solution
            outputFile << "[X,Y] = meshgrid(0:0.1:1, 0:0.1:1);" << endl;

            // Write the command to evaluate the solution at each point on the grid
            outputFile << "Z = x(1)*X + x(2)*Y + x(3);" << endl;

            // Write the command to plot the solution
            outputFile << "figure;" << endl;
            outputFile << "surf(X,Y,Z);" << endl;
            outputFile << "xlabel('x');" << endl;
            outputFile << "ylabel('y');" << endl;
            outputFile << "zlabel('z');" << endl;
            outputFile << "colorbar;" << endl;
        }
            outputFile.close();
    }
}// LinearAlgebra