//
// Created by hal9000 on 12/22/22.
//

#include "MeshTest2D.h"
#include "MeshPreProcessor.h"

namespace StructuredMeshGenerator {
    
        MeshTest2D::MeshTest2D() {
            //MeshSpecs specs = MeshSpecs(5, 5, 1.0, 1.0, 0.0, 0.0, 0.0);
            //MeshPreProcessor preProcessor = MeshPreProcessor(specs);
            
/*            NodeFactory nodeFactory = NodeFactory(specs.nnx, specs.nny, 0);
            for (int i = 0; i < specs.nnx; ++i) {
                for (int j = 0; j < specs.nny; ++j) {
                    Node *node = nodeFactory.nodesMatrix->element(i, j);
                    cout << "i : " << i << " j : " << j << " gl: " << *node->id->global
                         << " b: " << *node->id->boundary
                         << " in: " << *node->id->internal << endl;
                }
            }*/
        }
    } // StructuredMeshGenerator