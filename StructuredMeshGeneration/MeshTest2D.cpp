//
// Created by hal9000 on 12/22/22.
//

#include "MeshTest2D.h"


namespace StructuredMeshGenerator {
    
        MeshTest2D::MeshTest2D() {
            map<Direction, unsigned> numberOfNodes;
            numberOfNodes[Direction::One] = 5;
            numberOfNodes[Direction::Two] = 5;
            numberOfNodes[Direction::Three] = 1;    
            MeshSpecs specs = MeshSpecs(numberOfNodes, 1, 1, 0, 0, 0);
            auto space = (PositioningInSpace::OneTwo_plane);
            auto mesh = MeshPreProcessor(specs, reinterpret_cast<PhysicalSpaceEntity &>(space)).mesh;
            
            for(int j = 0; j < mesh->numberOfNodesPerDirection.at(Direction::Two); j++){
                for(int i = 0; i < mesh->numberOfNodesPerDirection.at(Direction::One); i++){
                    cout << "(i,j): (" << i <<" " << j << ")" <<  endl;
                    cout << "Global id: " << *(mesh->node(i,j)->id.global) << endl;
                    cout << "Boundary id: " << *(mesh->node(i,j)->id.boundary) << endl;
                    cout << "Internal id: " << *(mesh->node(i,j)->id.internal) << endl;
                    cout << "-------------" << endl;
                }
            }
            
            cout << "MTSTK GMS" << endl;
            
            
        }
    } // StructuredMeshGenerator