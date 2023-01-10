//
// Created by hal9000 on 11/28/22.
//

#include <iostream>
#include <list>
#include "Node.h"
#include "../Id/DiscreteEntityId.h"
using namespace Discretization;

namespace Discretization
{
    Node::Node(PhysicalSpaceEntity spaceEntity) {
        space = spaceEntity;
    }

    vector<double> Node::positionVector(PhysicalSpaceEntities physicalSpace) {
        return _position.getCoordinateVectorInEntity(space.type(), physicalSpace);
    }
    
    vector<double> Node::positionVector() {
        return _position.getCoordinateVectorIn3D(space.type());
    }
    
} // Discretization
    

