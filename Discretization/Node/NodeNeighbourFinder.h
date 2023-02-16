//
// Created by hal9000 on 1/22/23.
//

#ifndef UNTITLED_NODENEIGHBOURFINDER_H
#define UNTITLED_NODENEIGHBOURFINDER_H
#include <map>
#include "../../PositioningInSpace/DirectionsPositions.h"
using namespace std;
using namespace PositioningInSpace;

namespace Discretization {

    class NodeNeighbourFinder {
        public:
            map<Position, unsigned> neighbourMap(unsigned &nodeId, unsigned &nn1, unsigned &nn2, unsigned &totalNodes);
        private:
            map<Position, unsigned> _possibleNeighbours;
    };

} // Discretization

#endif //UNTITLED_NODENEIGHBOURFINDER_H
