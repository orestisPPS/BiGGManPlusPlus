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

        cout << "Coordinates: " << endl;
        cout << "---------------------------------------------------------" << endl;
        
/*        cout << "Natural coordinates [x_1, x_2, x_3] = { " <<(\
             coordinates(Natural, 0)) << ", " << coordinates(Natural, 1) << ", " << coordinates(Natural, 2) << "}" << endl;*/
                     
        auto parametricCoords = coordinates.positionVector3D(Parametric);
        cout << "Parametric coordinates [x_1, x_2, x_3] = { " <<(\
             parametricCoords[0]) << ", " << parametricCoords[1] << ", " << parametricCoords[2] << "}" << endl;
                     
        auto templateCoords = coordinates.positionVector3D(Template);
        cout << "Template coordinates [x_1, x_2, x_3] = { " <<(\
             templateCoords[0]) << ", " << templateCoords[1] << ", " << templateCoords[2] << "}" << endl;
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
    

