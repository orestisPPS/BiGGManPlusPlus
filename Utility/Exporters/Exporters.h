//
// Created by hal9000 on 4/6/23.
//

#ifndef UNTITLED_EXPORTERS_H
#define UNTITLED_EXPORTERS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../../LinearAlgebra/Array/Array.h"
#include "../../Discretization/Mesh/Mesh.h"

using namespace std;
using namespace LinearAlgebra;

namespace Utility {

    class Exporters {
        
    public:

        // Creates a .m file that can be takes as input the linear system matrix and vector and solves the system.
        // Then the solution is plotted.
        static void exportLinearSystemToMatlabFile(Array<double>* matrix, vector<double>* vector, const string& filePath, 
                                                   const string& fileName, bool print = false);

        static void exportLinearSystemToMatlabFile(Array<double> matrix, vector<double> vector, const string& filePath,
                                                   const string& fileName,  bool print = false);

        static void exportScalarFieldResultInVTK(const std::string& filePath, const std::string& fileName,
                                                 const std::string& fieldName, Mesh* mesh) {
            ofstream outputFile(filePath + fileName);
            outputFile << "# vtk DataFile Version 3.0 \n";
            outputFile << "vtk output \n";
            outputFile << "ASCII \n";
            outputFile << "DATASET UNSTRUCTURED_GRID \n";
            outputFile << "POINTS " << mesh->totalNodesVector->size() << " double\n";
            for (auto &node: *mesh->totalNodesVector) {
                auto coordinates = node->coordinates.positionVector3D(Natural);
                outputFile << coordinates[0] << " " << coordinates[1] << " " << coordinates[2] << "\n";
            }

            // Add field values
            outputFile << "\nPOINT_DATA " << mesh->totalNodesVector->size() << "\n";
            outputFile << "SCALARS " << fieldName << " double 1\n";  // Name and type of the scalar field.
            outputFile << "LOOKUP_TABLE default\n";
            for (auto &node: *mesh->totalNodesVector) {
                if (!node->degreesOfFreedom->empty()) {
                    outputFile << node->degreesOfFreedom->front()->value() << "\n";
                } else {
                    // Decide what to do if there are no degrees of freedom at a node. Maybe write out a default value.
                    outputFile << 0.0 << "\n";
                }
            }
            outputFile.close();
        }
        
        // Creates a .vtk file that can be opened in Paraview to visualize the mesh.
        //static void saveNodesToParaviewFile(Mesh* mesh, const std::string &filePath, const std::string &fileName);
        
        //static void saveGhostNodesToParaviewFile(GhostPseudoMesh* mesh, const std::string &filePath, const std::string &fileName);
    };

} // Utility

#endif //UNTITLED_EXPORTERS_H
