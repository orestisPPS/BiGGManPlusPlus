//
// Created by hal9000 on 1/22/23.
//

#ifndef UNTITLED_NODENEIGHBOURFINDER_H
#define UNTITLED_NODENEIGHBOURFINDER_H
#include <map>
#include "../../PositioningInSpace/DirectionsPositions.h"
#include "../../DegreesOfFreedom/DegreeOfFreedomTypes.h"
#include "../Mesh/Mesh.h"
using namespace DegreesOfFreedom;

using namespace std;
using namespace PositioningInSpace;

namespace Discretization {

    class NodeNeighbourFinder {
        public:
            //Returns a map of all the neighbour nodes of the given node id.
            //Key: Position Enum
            //Value: Node pointer
            static map<Position, Node*> getNeighbourNodes(Mesh* mesh, const unsigned* nodeId);
            
            //Returns a map of all the degrees of freedom of the neighbour nodes of the given node id.
            //Key: Position Enum
            //Value: Vector pointer of DOF id pointers of the neighbour node
            static map<Position, vector<unsigned*>*> getAllNeighbourDOF(Mesh* mesh, const unsigned* nodeId);
            
            //Returns a map of a specific degree of freedom of the neighbour nodes of the given node id.
            //Key: Position Enum
            //Value: DOF id pointer of dofType of the neighbour node
            static map<Position, unsigned*> getSpecificNeighbourDOF(Mesh* mesh, unsigned* nodeId, DOFType dofType);
        private:

    };

} // Discretization

#endif //UNTITLED_NODENEIGHBOURFINDER_H
