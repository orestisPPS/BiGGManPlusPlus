//
// Created by hal9000 on 4/6/23.
//

#ifndef UNTITLED_EXPORTERS_H
#define UNTITLED_EXPORTERS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../Discretization/Mesh/Mesh.h"

using namespace std;
using namespace LinearAlgebra;

namespace Utility {

    class Exporters {
        
    public:


        static void exportScalarFieldResultInVTK(const std::string& filePath, const std::string& fileName,
                                                 const std::string& fieldName, shared_ptr<Mesh> mesh) {
            ofstream outputFile(filePath + fileName);

            // Header
            outputFile << "# vtk DataFile Version 3.0\n";
            outputFile << "ParaView Output\n";
            outputFile << "ASCII\n";
            outputFile << "DATASET STRUCTURED_GRID\n";

            // Assuming the mesh is nx x ny x nz, specify the dimensions
            unsigned int nx = mesh->nodesPerDirection.at(One);
            unsigned int ny = mesh->nodesPerDirection.at(Two);
            unsigned int nz = mesh->nodesPerDirection.at(Three);
            outputFile << "DIMENSIONS " << nx << " " << ny<< " " << nz<< "\n";

            // Points
            outputFile << "POINTS " << mesh->totalNodesVector->size() << " double\n";
            for (auto &node : *mesh->totalNodesVector) {
                auto coordinates = node->coordinates.positionVector3D(Natural);
                outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << "\n";
            }

            // Add field values
            outputFile << "\nPOINT_DATA " << mesh->totalNodesVector->size() << "\n";
            outputFile << "SCALARS " << fieldName << " double\n";  // Name and type of the scalar field.
            outputFile << "LOOKUP_TABLE default\n";
            for (auto &node : *mesh->totalNodesVector) {
                if (!node->degreesOfFreedom->empty()) {
                    outputFile << node->degreesOfFreedom->front()->value() << "\n";
                } else {
                    // Use a default value for nodes without degrees of freedom.
                    outputFile << 0.0 << "\n";
                }
            }
            outputFile.close();
        }



        // Creates a .vtk file that can be opened in Paraview to visualize the mesh.
        //static void saveNodesToParaviewFile(shared_ptr<Mesh> mesh, const std::string &filePath, const std::string &fileName);
        
        //static void saveGhostNodesToParaviewFile(GhostPseudoMesh* mesh, const std::string &filePath, const std::string &fileName);
        static void
        exportLinearSystemToMatlabFile(const shared_ptr<NumericalMatrix<double>>& matrix, const shared_ptr<NumericalVector<double>>& vector,
                                       const std::string& filePath, const std::string& fileName, bool print);

        static void exportMatrixToMatlabFile(const shared_ptr<NumericalMatrix<double>>& matrix,
        const std::string& filePath, const std::string& fileName, bool print);

    };

} // Utility

#endif //UNTITLED_EXPORTERS_H
