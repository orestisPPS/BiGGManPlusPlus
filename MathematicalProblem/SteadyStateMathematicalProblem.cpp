//
// Created by hal9000 on 12/17/22.
//

#include "SteadyStateMathematicalProblem.h"
namespace MathematicalProblem{
    
    SteadyStateMathematicalProblem::SteadyStateMathematicalProblem(PartialDifferentialEquation *pde,
                                                                   map<Position,list<BoundaryConditions::BoundaryCondition*>*> *bcs,
                                                                   list<DegreeOfFreedom*> *dof,
                                                                   SpaceEntityType space)
    : pde(pde), boundaryConditions(bcs), degreesOfFreedom(dof), space(space){
        checkBoundaryConditions();
        checkDegreesOfFreedom();
        checkSpaceEntityType();
    }

    SteadyStateMathematicalProblem::~SteadyStateMathematicalProblem() {
/*      delete pde;
        delete boundaryConditions;
        delete degreesOfFreedom;*/
    }
    
    void SteadyStateMathematicalProblem::checkBoundaryConditions() const {
        if (boundaryConditions->empty()){
            throw invalid_argument ("No boundary conditions were specified.");
        }
        if (space == SpaceEntityType::Axis &&
        (boundaryConditions->find(Position::Left) != boundaryConditions->end() ||
         boundaryConditions->find(Position::Right) != boundaryConditions->end())){
            throw invalid_argument("Boundary Conditions for 1D problems should be specified at the left"
                                   " and / or the left boundaries.");
        }
        
        if (space == SpaceEntityType::Plane &&
            (boundaryConditions->find(Position::Left) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Right) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Bottom) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Top) != boundaryConditions->end())){
            throw invalid_argument("Boundary Conditions for 2D problems should be specified at at least one of the "
                                   "following Directions: Left, Right, Bottom, Top.");
        }
        
        if (space == SpaceEntityType::Volume &&
            (boundaryConditions->find(Position::Left) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Right) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Bottom) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Top) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Front) != boundaryConditions->end() ||
             boundaryConditions->find(Position::Back) != boundaryConditions->end())){
            throw invalid_argument("Boundary Conditions for 3D problems should be specified at at least one of the "
                                   "following Directions: Left, Right, Bottom, Top, Front, Back.");
        }
    }
    
    void SteadyStateMathematicalProblem::checkDegreesOfFreedom() const {
        //TODO : Implement this method
    }
    
    void SteadyStateMathematicalProblem::checkSpaceEntityType() const {
        if (space == PositioningInSpace::NullSpace){
            throw invalid_argument("The void inside you cannot be solved. Like this PDE you try to solve on a"
                                   "null space.");
        }
    }
}
