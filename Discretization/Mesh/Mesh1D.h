//
// Created by hal9000 on 3/11/23.
//

#ifndef UNTITLED_MESH1D_H
#define UNTITLED_MESH1D_H

#include "Mesh.h"
#include "GhostPseudoMesh/GhostPseudoMesh.h"

namespace Discretization {

    class Mesh1D : public Mesh {
    public:
        Mesh1D(shared_ptr<Array<Node*>>nodes);
        
        ~Mesh1D();

        unsigned dimensions() override;
        
        unsigned numberOfInternalNodes() override;
        
        SpaceEntityType space() override;
        
        vector<Direction> directions() override;
                
        Node* node(unsigned i) override;
        
        Node* node(unsigned i, unsigned j) override;
        
        Node* node(unsigned i, unsigned j, unsigned k) override;
        
        void printMesh() override;
        
        NumericalVector<double> getNormalUnitVectorOfBoundaryNode(Position boundaryPosition, Node *node) override;

        unique_ptr<vector<Node*>> getInternalNodesVector() override;
        
        void createElements(ElementType elementType, unsigned int nodesPerEdge) override;
        
        void storeMeshInVTKFile(const string& filePath, const string& fileName,
                                CoordinateType coordinateType = Natural, bool StoreOnlyNodes = false) const override;
        
        
        
    protected:
        
        void _addDBoundaryNodesToMap() override;
        
        void _addTotalNodesToVector() override;
        
        //GhostPseudoMesh* _createGhostPseudoMesh(unsigned ghostLayerDepth) override;
        
    };

} // Discretization

#endif //UNTITLED_MESH1D_H
