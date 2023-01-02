//
// Created by hal9000 on 12/22/22.
//

#include "MeshTest2D.h"

namespace StructuredMeshGenerator {
    
        MeshTest2D::MeshTest2D() {
            MeshSpecs2D specs = MeshSpecs2D(5, 5, 1.0, 1.0, 0.0, 0.0, 0.0);
            NodeFactory nodeFactory = NodeFactory(specs.nnx, specs.nny, 0);
            for (int i = 0; i < specs.nnx; ++i) {
                for (int j = 0; j < specs.nny; ++j) {
                    Node* node = nodeFactory.nodesMatrix->element(i, j);
                    cout <<"i : " << i << " j : " << j << " Node " << node->id ->global   << endl;
                }
            }
        }
    } // StructuredMeshGenerator