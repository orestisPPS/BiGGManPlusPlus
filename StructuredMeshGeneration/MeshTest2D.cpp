//
// Created by hal9000 on 12/22/22.
//

#include "MeshTest2D.h"


namespace StructuredMeshGenerator {
    
        MeshTest2D::MeshTest2D() {
            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 5;
            MeshSpecs specs = MeshSpecs(numberOfNodes, 1, 1, 0, 0, 0);
            auto space = (PositioningInSpace::Plane);
            auto mesh = MeshFactory(specs).mesh;
            mesh->printMesh();
            
            cout << "MTSTK GMS" << endl;
        }
    } // StructuredMeshGenerator