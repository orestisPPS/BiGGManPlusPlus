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
            //Print all the nodes and all their ids and coordinates
            for (unsigned i = 0; i < mesh->numberOfNodesPerDirection.at(Direction::Two); ++i) {
                for (unsigned j = 0; j < mesh->numberOfNodesPerDirection.at(Direction::One); ++j) {
                    cout << "(i,j): (" << j <<" " << i << ")" <<  endl;
                    cout << "Global id: " << *(mesh->node(j, i)->id.global) << endl;
                    cout << "Boundary id: " << *(mesh->node(j, i)->id.boundary) << endl;
                    cout << "Internal id: " << *(mesh->node(j, i)->id.internal) << endl;
                    cout << "-------------" << endl;
                    //cout << "Node coordinates: " << *(mesh->node(j, i)->positionVector(Natural)[0]) << " " << mesh->node(j, i)->positionVector(Natural)[1]) << endl;
                    //cout << "Node coordinates: " << mesh->node(i, j)->positionVector(Natural)[0], mesh->node(i, j)->positionVector(Natural)[0] << endl;
                }
            }
            cout << "MTSTK GMS" << endl;
            
            
        }
    } // StructuredMeshGenerator