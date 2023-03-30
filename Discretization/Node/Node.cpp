//
// Created by hal9000 on 11/28/22.
//

#include "Node.h"

namespace Discretization {
    Node::Node() : id(DiscreteEntityId()), coordinates() {
        degreesOfFreedom = new vector<DegreeOfFreedom*>();
    }
    
    void Node::printNode() {
        cout << "Node: " << (*id.global) << endl;
        cout << "Boundary value: " << (*id.boundary) << " Internal Id: "<< (*id.internal)<< endl;
        cout << "Node coordinates [x_1, x_2, x_3] = { " <<(coordinates.positionVector(Template)[0]) << ", " <<
                                                    coordinates.positionVector(Template)[1] << ", " <<
                                                    coordinates.positionVector(Template)[2] << "}" << endl;
        cout << "-------------------------------------------" << endl;
    }
}// Discretization
    

