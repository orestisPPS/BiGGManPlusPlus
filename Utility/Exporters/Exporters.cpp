//
// Created by hal9000 on 4/6/23.
//

#include "Exporters.h"

namespace Utility {
    //Author: Chat GPT
    void Exporters::exportLinearSystemToMatlabFile(const shared_ptr<NumericalMatrix<double>>& matrix, const shared_ptr<NumericalVector<double>>&vector, const std::string& filePath,
                                                   const std::string& fileName, bool print) {
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
        for (double i : *vector) {
            outputFile << i << " ";
        }
        outputFile << "]';" << endl;

        // Write the command to solve the system
        outputFile << "x = A \\ b;" << endl;
        outputFile <<"eig(A)" << endl;
    }

    void Exporters::exportMatrixToMatlabFile(const shared_ptr<NumericalMatrix<double>> &matrix, const string &filePath,
                                             const string &fileName, bool print) {
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
        
        outputFile << "eig(A)" << endl;
        outputFile.close();
    }

/*    void Exporters::saveNodesToParaviewFile(shared_ptr<Mesh> mesh, const std::string& filePath, const std::string& fileName) {
        ofstream outputFile(filePath + fileName);
        outputFile << "# vtk DataFile Version 3.0 \n";
        outputFile << "vtk output \n" ;
        outputFile << "ASCII \n" ;
        outputFile << "DATASET UNSTRUCTURED_GRID \n";
        outputFile << "POINTS " << mesh->totalNodesVector->size() << " double\n" ;
        for (auto &node: *mesh->totalNodesVector) {
            auto coordinates = node->coordinates.positionVector(Template);
            outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << "\n" ;
        }

*//*        for (auto &node: *mesh->totalNodesVector) {
            auto coordinates = node->coordinates.positionVector(Parametric);
            outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << endl;
        }

        for (auto &node: *mesh->totalNodesVector) {
            auto coordinates = node->coordinates.positionVector(Template);
            outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << endl;
        }*//*

        outputFile.close();
    }
    
    void Exporters::saveGhostNodesToParaviewFile(GhostPseudoMesh *mesh, const std::string& filePath, const std::string& fileName) {
*//*        ofstream outputFile(filePath + fileName);
        outputFile << "# vtk DataFile Version 3.0 \n";
        outputFile << "vtk output \n" ;
        outputFile << "ASCII \n" ;
        outputFile << "DATASET UNSTRUCTURED_GRID \n";
        outputFile << "POINTS " << mesh->allNodesList->size() << " double\n" ;
        for (auto &node: *mesh->allNodesList) {
            auto coordinates = node->coordinates.positionVector(Template);
            outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << "\n" ;
        }
        outputFile.close();*//*
    }
    
    */
    
} // Utility