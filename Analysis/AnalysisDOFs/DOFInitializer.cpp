//
// Created by hal9000 on 2/18/23.
//

#include <algorithm>
#include "DOFInitializer.h"

namespace NumericalAnalysis {
    
    DOFInitializer::DOFInitializer(Mesh* mesh, DomainBoundaryConditions* domainBoundaryConditions, Field_DOFType* degreesOfFreedom) {
        freeDegreesOfFreedom = new list<DegreeOfFreedom*>();
        boundedDegreesOfFreedom = new list<DegreeOfFreedom*>();
        fluxDegreesOfFreedom = new list<tuple<DegreeOfFreedom*, double>>;
        totalDegreesOfFreedom = new list<DegreeOfFreedom*>();

        initiateInternalNodeDOFs(mesh, degreesOfFreedom);
        initiateBoundaryNodeDOF(mesh, degreesOfFreedom, domainBoundaryConditions);
        removeDuplicatesAndDelete(mesh);
        assignDOFIDs();
        //printDOFList(totalDegreesOfFreedom);
        
    }
    
    void DOFInitializer::initiateInternalNodeDOFs(Mesh *mesh, Field_DOFType *degreesOfFreedom) const {
                
        //March through the mesh nodes
        for (auto & internalNode : *mesh->internalNodes){
            for (auto & DOFType : *degreesOfFreedom->DegreesOfFreedom){
                auto dof = new DegreeOfFreedom(DOFType, internalNode->id.global, false);
                freeDegreesOfFreedom->push_back(dof);
                totalDegreesOfFreedom->push_back(dof);
            }            
        }
    }
    void DOFInitializer::initiateBoundaryNodeDOF(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                     DomainBoundaryConditions *domainBoundaryConditions) const {
        for (auto & domainBoundary : *mesh->boundaryNodes){
            auto position = domainBoundary.first;
            auto nodesAtPosition = domainBoundary.second;
            auto bcAtPosition = domainBoundaryConditions-> getBoundaryConditionAtPosition(position);
            //March through the nodes at the Position  that is the key of the domainBoundary map
            for (auto & node : *nodesAtPosition){
                auto dofValue = -1.0;
                for (auto& dofType : *problemDOFTypes->DegreesOfFreedom){
                    dofValue = bcAtPosition->scalarValueOfDOFAt((*dofType), node->coordinates.positionVectorPtr());
                    switch (bcAtPosition->type()){
                        case Dirichlet:{
                            auto dirichletDOF = new DegreeOfFreedom(dofType, dofValue, node->id.global, true);
                            boundedDegreesOfFreedom->push_back(dirichletDOF);
                            totalDegreesOfFreedom->push_back(dirichletDOF);
                            break;
                        }
                        case Neumann:{
                            auto neumannDOF = new DegreeOfFreedom(dofType, node->id.global, false);
                            fluxDegreesOfFreedom->push_back(tuple<DegreeOfFreedom*, double>(neumannDOF, dofValue));
                            freeDegreesOfFreedom->push_back(neumannDOF);
                            totalDegreesOfFreedom->push_back(neumannDOF);
                            break;
                        }
                        default:
                            throw std::invalid_argument("Boundary condition type not recognized");
                    }
                }
            }
        }
    }
    
    
    void DOFInitializer::removeDuplicatesAndDelete(Mesh *mesh) const {
        
        //Sort the list of total DOFs
        totalDegreesOfFreedom->sort([](DegreeOfFreedom* a, DegreeOfFreedom* b){
            return (*a->parentNode) < (*b->parentNode);
        });
        boundedDegreesOfFreedom->sort([&mesh](DegreeOfFreedom* a, DegreeOfFreedom* b){
            auto aNodeBoundaryID = (*mesh->nodeFromID((*a->parentNode))->id.boundary);
            auto bNodeBoundaryID = (*mesh->nodeFromID((*b->parentNode))->id.boundary);
            return aNodeBoundaryID > bNodeBoundaryID;
        });
        fluxDegreesOfFreedom->sort([](tuple<DegreeOfFreedom*, double> a, tuple<DegreeOfFreedom*, double> b){
            return (*get<0>(a)->parentNode) < (*get<0>(b)->parentNode);
        });
        freeDegreesOfFreedom->sort([](DegreeOfFreedom* a, DegreeOfFreedom* b){
            return (*a->parentNode) < (*b->parentNode);
        });
        
        auto duplicates = vector<DegreeOfFreedom*>();
        //Remove the duplicates from the list of total DOFs. The duplicates are added to the duplicates vector.
        totalDegreesOfFreedom->unique([&duplicates](DegreeOfFreedom* a, DegreeOfFreedom* b){
            auto parentNodeCondition = (*a->parentNode) == (*b->parentNode);
            auto dofTypeCondition = a->type() == b->type();
            auto constraintTypeCondition = false;
            //auto aDirichletBNeumann = a->id->constraintType() == ConstraintType::Fixed && b->id->constraintType() == ConstraintType::Free;

            constraintTypeCondition = a->id->constraintType() == Fixed && b->id->constraintType() == Free;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition){
                duplicates.push_back(b);
                return true;
            }
            constraintTypeCondition = a->id->constraintType() == Free && b->id->constraintType() == Fixed;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition){
                duplicates.push_back(a);
                return true;
            }
            constraintTypeCondition = a->id->constraintType() == Fixed && b->id->constraintType() == Fixed;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition){
                 duplicates.push_back(b);
                return true;
            }
            constraintTypeCondition = a->id->constraintType() == Free && b->id->constraintType() == Free;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition){
                duplicates.push_back(b);
                return true;
            }
            return false;
        });
        
        //Remove the duplicates of the duplicates vector from the bounded or free and flux DOF lists according to
        // the constraint type their value (fixed or free).
        for (auto &dof : duplicates) {
            if (dof->id->constraintType() == ConstraintType::Fixed) {
                boundedDegreesOfFreedom->remove(dof);
            }
            else if (dof->id->constraintType() == ConstraintType::Free) {
                //FluxDOFs list has tuples of DOF and flux value. Delete the DOF from the list only from the first item of the tuple
                for (auto &fluxDOF : *fluxDegreesOfFreedom) {
                    if (get<0>(fluxDOF) == dof) {
                        fluxDegreesOfFreedom->remove(fluxDOF);
                        break;
                    }
                }
                freeDegreesOfFreedom->remove(dof);
            }
            delete dof;
            dof = nullptr;
        }
    }

    void DOFInitializer::assignDOFIDs() const {
        unsigned dofID = 0;
        for (auto &dof : *freeDegreesOfFreedom) {
            (*dof->id->value) = dofID;
            cout << "FREE DOF ID: " << dofID << " Node: " << (*dof->parentNode)<< endl;
            dofID++;
        }
        cout <<" "<< endl;
        dofID = 0;
        for (auto &dof : *boundedDegreesOfFreedom) {
            (*dof->id->value) = dofID;
            cout << "FIXED DOF ID: " << dofID << " Node: " << (*dof->parentNode)<< endl;
            dofID++;
        }
    }
}

        