//
// Created by hal9000 on 1/22/23.
//

#include "NodeNeighbourFinder.h"

namespace Discretization {
    
    map<Position, Node*>
    NodeNeighbourFinder::getNeighbourNodes(Mesh *mesh, const unsigned* nodeId) {
        auto id = *nodeId;

        auto nn1 = mesh->numberOfNodesPerDirection[One];
        auto nn2 = mesh->numberOfNodesPerDirection[Two];
        auto nn3 = mesh->numberOfNodesPerDirection[Three];

        auto maxId = nn1 * nn2 * nn3 - 1;
        auto k = id / (nn1 * nn2);
        auto j = (id - k * nn1 * nn2) / nn1;
        auto i = id - k * nn1 * nn2 - j * nn1;
        
        map<Position, Node*> neighbours;
        // Central slice of a cube that includes the central node
                
        if (id + nn1 - 1 <= maxId) neighbours[TopLeft] =  mesh->nodeFromID(id + nn1 - 1);
        if (id + nn1 <= maxId) neighbours[Top] =  mesh->nodeFromID(id + nn1);
        if (id + nn1 + 1 <= maxId) neighbours[TopRight] =  mesh->nodeFromID(id + nn1 + 1);
        if (id - 1 <= maxId) neighbours[Left] =  mesh->nodeFromID(id - 1);
        if (id + 1 <= maxId) neighbours[Right] =  mesh->nodeFromID(id + 1);
        if (id - nn1 - 1 <= maxId) neighbours[BottomLeft] =  mesh->nodeFromID(id - nn1 - 1);
        if (id - nn1 <= maxId) neighbours[Bottom] =  mesh->nodeFromID(id - nn1);
        if (id - nn1 + 1 <= maxId) neighbours[BottomRight] =  mesh->nodeFromID(id - nn1 + 1);

        // Front slice of a cube
        if (k > 0) {
            if (id - nn2 - nn1 - 1 <= maxId) neighbours[FrontTopLeft] =  mesh->nodeFromID(id - nn2 - nn1 - 1);
            if (id - nn2 - nn1 <= maxId) neighbours[FrontTop] =  mesh->nodeFromID(id - nn2 - nn1);
            if (id - nn1 - nn2 + 1 <= maxId) neighbours[FrontTopRight] =  mesh->nodeFromID(id - nn1 - nn2 + 1);
            if (id - 2 * nn1 - nn2 - 1 <= maxId) neighbours[FrontLeft] =  mesh->nodeFromID(id - 2 * nn1 - nn2 - 1);
            if (id - 2 * nn1 - nn2 <= maxId) neighbours[Front] =  mesh->nodeFromID(id - 2 * nn1 - nn2);
            if (id - 2 * nn1 - nn2 + 1 <= maxId) neighbours[FrontRight] =  mesh->nodeFromID(id - 2 * nn1 - nn2 + 1);
            if (id - 3 * nn1 - nn2 - 1 <= maxId) neighbours[FrontBottomLeft] =  mesh->nodeFromID(id - 3 * nn1 - nn2 - 1);
            if (id - 3 * nn1 - nn2 <= maxId) neighbours[FrontBottom] =  mesh->nodeFromID(id - 3 * nn1 - nn2);
            if (id - 3 * nn1 - nn2 + 1 <= maxId) neighbours[FrontBottomRight] =  mesh->nodeFromID(id - 3 * nn1 - nn2 + 1);
        }

        // Back slice of a cube
        if (k < nn3 - 1) {
            if (id + 3 * nn1 + nn2 - 1 <= maxId) neighbours[BackTopLeft] =  mesh->nodeFromID(id + 3 * nn1 + nn2 - 1);
            if (id + 3 * nn1 + nn2 <= maxId) neighbours[BackTop] =  mesh->nodeFromID(id + 3 * nn1 + nn2);
            if (id + 3 * nn1 + nn2 + 1 <= maxId) neighbours[BackTopRight] =  mesh->nodeFromID(id + 3 * nn1 + nn2 + 1);
            if (id + 2 * nn1 + nn2 - 1 <= maxId) neighbours[BackLeft] =  mesh->nodeFromID(id + 2 * nn1 + nn2 - 1);
            if (id + 2 * nn1 + nn2 <= maxId) neighbours[Back] =  mesh->nodeFromID(id + 2 * nn1 + nn2);
            if (id + 2 * nn1 + nn2 + 1 <= maxId) neighbours[BackRight] =  mesh->nodeFromID(id + 2 * nn1 + nn2 + 1);
            if (id + nn1 + nn2 - 1 <= maxId) neighbours[BackBottomLeft] =  mesh->nodeFromID(id + nn1 + nn2 - 1);
            if (id + nn1 + nn2 <= maxId) neighbours[BackBottom] =  mesh->nodeFromID(id + nn1 + nn2);
            if (id + nn1 + nn2 + 1 <= maxId) neighbours[BackBottomRight] =  mesh->nodeFromID(id + nn1 + nn2 + 1);
        }
        
        //Remove the neighbours that are outside the mesh
        //TODO: remove later if all the above conditions are correct
        for (auto nullNeighbour = neighbours.begin(); nullNeighbour != neighbours.end(); nullNeighbour++) {
            if (nullNeighbour->second == nullptr) {
                neighbours.erase(nullNeighbour);
            }
        }
        
        return neighbours;
    }
    
    map<Position, vector<DegreeOfFreedom*>*>
    NodeNeighbourFinder::getAllNeighbourDOF(Mesh *mesh, const unsigned int *nodeId) {
        auto neighbours = getNeighbourNodes(mesh, nodeId);
        map<Position, vector<DegreeOfFreedom*>*> neighbourDOF;
        for (auto & neighbour : neighbours) {
            neighbourDOF[neighbour.first] = neighbour.second->degreesOfFreedom;
        }
        return neighbourDOF;    
    }

    map<Position, DegreeOfFreedom*>
    NodeNeighbourFinder::getSpecificNeighbourDOF(Mesh *mesh, unsigned* nodeId, DOFType dofType) {
        auto dofHood = getAllNeighbourDOF(mesh, nodeId);
        map<Position, DegreeOfFreedom*> specificNeighbourDOF;
        for (auto & dofAtPosition : dofHood) {
            for (auto & dofVectorComponent : *dofAtPosition.second) {
                if (dofVectorComponent->type() == dofType) {
                    specificNeighbourDOF[dofAtPosition.first] = dofVectorComponent;
                }
            }
        }
        

    }
    
    
} // Discretization