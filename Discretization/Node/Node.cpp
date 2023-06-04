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
/*        //cout << "Boundary value: " << (*id.boundary) << " Internal Id: "<< (*id.internal)<< endl;
        cout << "Template coordinates [x_1, x_2, x_3] = { " <<(coordinates.positionVector(Template)[0]) << ", " <<
                                                           coordinates.positionVector(Template)[1] << ", " <<
                                                           coordinates.positionVector(Template)[2] << "}" << endl;
        cout << "Parametric coordinates [x_1, x_2, x_3] = { " <<(coordinates.positionVector(Parametric)[0]) << ", " <<
             coordinates.positionVector(Parametric)[1] << ", " <<
             coordinates.positionVector(Parametric)[2] << "}" << endl;*/
        auto naturalCoords = coordinates.positionVector3D(Natural);
        cout << "Natural coordinates [x_1, x_2, x_3] = { " <<(\
             naturalCoords[0]) << ", " << naturalCoords[1] << ", " << naturalCoords[2] << "}" << endl;
        cout << "---------------------------------------------------------" << endl;
    }
    
    DegreeOfFreedom* Node::getDegreeOfFreedomPtr(DOFType type) const {
        for (auto &dof : *degreesOfFreedom) {
            if (dof->type() == type) {
                return dof;
            }
        }
        return nullptr;
    }
    
    const DegreeOfFreedom& Node::getDegreeOfFreedom(DOFType type) const {
        for (auto &dof : *degreesOfFreedom) {
            if (dof->type() == type) {
                return *dof;
            }
        }
        return *degreesOfFreedom->at(0);
    }
}// Discretization
    

