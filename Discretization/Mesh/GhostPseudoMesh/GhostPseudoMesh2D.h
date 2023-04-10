//
// Created by hal9000 on 4/11/23.
//

#ifndef UNTITLED_GHOSTPSEUDOMESH2D_H
#define UNTITLED_GHOSTPSEUDOMESH2D_H

#include "GhostPseudoMesh.h"

namespace Discretization {

    class GhostPseudoMesh2D : public GhostPseudoMesh {

        GhostPseudoMesh2D(Mesh* targetMesh, map<Direction, unsigned>* ghostNodesPerDirection);

        ~GhostPseudoMesh2D();

        Array<Node*> *createGhostedNodesMatrix() override;

        Node* node(unsigned i) override;

        Node* node(unsigned i, unsigned j) override;

        Node* node(unsigned i, unsigned j, unsigned k) override;
    };

} // Discretization

#endif //UNTITLED_GHOSTPSEUDOMESH2D_H
