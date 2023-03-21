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
        
        initiateBoundaryNodeFixedDOF(mesh, degreesOfFreedom, domainBoundaryConditions);
        initiateInternalNodeDOFs(mesh, degreesOfFreedom);
        //initiateBoundaryNodeFluxDOF(mesh, degreesOfFreedom, domainBoundaryConditions);
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

    void DOFInitializer::initiateBoundaryNodeFluxDOF(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                                       DomainBoundaryConditions *domainBoundaryConditions) const {
        //March through the domainBoundarys map
        for (auto & domainBoundary : *mesh->boundaryNodes){
            auto position = domainBoundary.first;
            auto nodesAtPosition = domainBoundary.second;
            auto neumannBCAtPosition = domainBoundaryConditions->
                    GetBoundaryConditions(position, BoundaryConditionType::Neumann);
            //March through the nodes at the Position  that is the key of the domainBoundary map
            for (auto & node : *nodesAtPosition){
                
                auto fluxValue = -1.0;
                //If the length of the list of Neumann BCs at the position is 1,
                // the same value of flux is applied to all nodes of the boundary
                if (problemDOFTypes->DegreesOfFreedom->size() == 1 &&neumannBCAtPosition->size() == 1){
                    fluxValue = neumannBCAtPosition->front()->scalarValueAt(node->coordinates.positionVectorPtr());
                    auto dof = new DegreeOfFreedom(problemDOFTypes->DegreesOfFreedom->front(),
                                                   node->id.global, true);
                    fluxDegreesOfFreedom->push_back(tuple<DegreeOfFreedom*, double>(dof, fluxValue));
                    freeDegreesOfFreedom->push_back(dof);
                    totalDegreesOfFreedom->push_back(dof);
                }
                else if (problemDOFTypes->DegreesOfFreedom->size() > 1 &&neumannBCAtPosition->size() > 1){
                    auto fluxVectorValue = neumannBCAtPosition->front()->vectorValueAt(
                            node->coordinates.positionVectorPtr());
                    auto dofType = problemDOFTypes->DegreesOfFreedom->front();
                    for (int i = 0; i < problemDOFTypes->DegreesOfFreedom->size(); i++){
                        dofType = problemDOFTypes->DegreesOfFreedom->at(i);
                        auto dof = new DegreeOfFreedom(dofType,node->id.global, true);
                        fluxDegreesOfFreedom->push_back(tuple<DegreeOfFreedom*, double>(dof, fluxVectorValue[i]));
                        freeDegreesOfFreedom->push_back(dof);
                        totalDegreesOfFreedom->push_back(dof);
                    }
                }
                else{
                    throw std::invalid_argument("The number of DOF types and the number of Neumann BCs at the boundary are not equal.");
                }
            }
        }
    }
    
    void printDOFList(list<DegreeOfFreedom*> *dofList){
        for (auto & dof : *dofList){
            cout << (*dof->parentNode) << endl;
        }
    }

    void DOFInitializer::initiateBoundaryNodeFixedDOF(Mesh *mesh, Field_DOFType *degreesOfFreedom,
                                                                        DomainBoundaryConditions *domainBoundaryConditions) const {
        //March through the boundaryNodes map
        for (auto &domainBoundary: *mesh->boundaryNodes) {
            auto position = domainBoundary.first;
            auto nodesAtPosition = domainBoundary.second;
            auto dirichletBCAtPosition = domainBoundaryConditions->
                    GetBoundaryConditions(position, BoundaryConditionType::Dirichlet);
            //March through the nodes at the Position  that is the key of the domainBoundary map
            for (auto &node: *nodesAtPosition) {
                auto dofValue = -1.0;
                //If the length of the list of Dirichlet BCs at the position is 1,
                // the same value of flux is applied to all nodes of the boundary
                if (degreesOfFreedom->DegreesOfFreedom->size() == 1 && dirichletBCAtPosition->size() == 1) {
                    dofValue = dirichletBCAtPosition->front()->scalarValueAt(
                            node->coordinates.positionVectorPtr());
                    auto dof = new DegreeOfFreedom(degreesOfFreedom->DegreesOfFreedom->front(), dofValue,
                                                   node->id.global, true);
                    boundedDegreesOfFreedom->push_back(dof);
                    totalDegreesOfFreedom->push_back(dof);
                }
                
                else if (degreesOfFreedom->DegreesOfFreedom->size() > 1 && dirichletBCAtPosition->size() > 1) {
                    auto vectorValue = dirichletBCAtPosition->front()->vectorValueAt(
                            node->coordinates.positionVectorPtr());
                    auto dofType = degreesOfFreedom->DegreesOfFreedom->front();
                    for (int i = 0; i < degreesOfFreedom->DegreesOfFreedom->size(); i++) {
                        dofValue = vectorValue[i];
                        dofType = degreesOfFreedom->DegreesOfFreedom->at(i);
                    }
                    auto dof = new DegreeOfFreedom(dofType, dofValue, node->id.global, true);
                    boundedDegreesOfFreedom->push_back(dof);
                    totalDegreesOfFreedom->push_back(dof);
                } else {
                    throw std::invalid_argument(
                            "The number of DOF types and the number of Dirichlet BCs at the boundary are not equal.");
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
            if ((*a->parentNode) == (*b->parentNode)){
                duplicates.push_back(b);
            }
            return (*a->parentNode) == (*b->parentNode);
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

        