//
// Created by hal9000 on 4/11/23.
//

#ifndef UNTITLED_GHOSTPSEUDOMESH3D_H
#define UNTITLED_GHOSTPSEUDOMESH3D_H

#include "GhostPseudoMesh.h"
#include "../../../LinearAlgebra/Transformations.h"


namespace Discretization {

    class GhostPseudoMesh3D : public GhostPseudoMesh {

        GhostPseudoMesh3D(Mesh* targetMesh, map<Direction, unsigned>* ghostNodesPerDirection);

        ~GhostPseudoMesh3D();

        Array<Node*> *createGhostedNodesMatrix() override;

        Node* node(unsigned i) override;

        Node* node(unsigned i, unsigned j) override;

        Node* node(unsigned i, unsigned j, unsigned k) override;
    };

} // Discretization

#endif //UNTITLED_GHOSTPSEUDOMESH3D_H
