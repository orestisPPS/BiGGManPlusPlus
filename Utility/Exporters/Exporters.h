//
// Created by hal9000 on 4/6/23.
//

#ifndef UNTITLED_EXPORTERS_H
#define UNTITLED_EXPORTERS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../../LinearAlgebra/Array.h"
//#include "../../Discretization/Mesh/Mesh.h"

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
        
        // Creates a .vtk file that can be opened in Paraview to visualize the mesh.
        //static void saveNodesToParaviewFile(Mesh* mesh, const std::string &filePath, const std::string &fileName);
        
        //static void saveGhostNodesToParaviewFile(GhostPseudoMesh* mesh, const std::string &filePath, const std::string &fileName);
    };

} // Utility

#endif //UNTITLED_EXPORTERS_H
