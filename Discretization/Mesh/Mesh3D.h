//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH3D_H
#define UNTITLED_MESH3D_H

#include "Mesh.h"

namespace Discretization {

    class Mesh3D : public Mesh {
    public:
        Mesh3D(shared_ptr<Array<Node*>>nodes);

        ~Mesh3D();

        unsigned dimensions() override;

        unsigned numberOfInternalNodes() override;

        SpaceEntityType space() override;

        vector<Direction> directions() override;

        unique_ptr<vector<Node*>> getInternalNodesVector() override;

        
        Node* node(unsigned i) override;

        Node* node(unsigned i, unsigned j) override;

        Node* node(unsigned i, unsigned j, unsigned k) override;
        
        void printMesh() override;
        
        NumericalVector<double> getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) override;
        
        void createElements(ElementType elementType, unsigned int nodesPerEdge) override;

        void storeMeshInVTKFile(const string& filePath, const string& fileName,
                                CoordinateType coordinateType = Natural, bool StoreOnlyNodes = false) const override;
        
    protected:
        
        shared_ptr<map<Position, shared_ptr<vector<Node*>>>>_addDBoundaryNodesToMap() override;
        
        shared_ptr<vector<Node*>> _addTotalNodesToVector() override;
        
        //GhostPseudoMesh* _createGhostPseudoMesh(unsigned ghostLayerDepth) override;
    };

} // Discretization

#endif //UNTITLED_MESH3D_H
