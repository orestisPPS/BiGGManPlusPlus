//
// Created by hal9000 on 2/18/23.
//

#include <algorithm>
#include "DOFInitializer.h"

namespace NumericalAnalysis {
    
    DOFInitializer::DOFInitializer(Mesh* mesh, DomainBoundaryConditions* domainBoundaryConditions, Field_DOFType* degreesOfFreedom) {
        
        _freeDegreesOfFreedomList = new list<DegreeOfFreedom*>();
        freeDegreesOfFreedom =  new vector<DegreeOfFreedom*>();
        _boundedDegreesOfFreedomList = new list<DegreeOfFreedom*>();
        boundedDegreesOfFreedom = new vector<DegreeOfFreedom*>();
        _fluxDegreesOfFreedomList = new list<tuple<DegreeOfFreedom*, double>>;
        fluxDegreesOfFreedom = new vector<tuple<DegreeOfFreedom*, double>>;
        _totalDegreesOfFreedomList = new list<DegreeOfFreedom*>();
        totalDegreesOfFreedom = new vector<DegreeOfFreedom*>();
        totalDegreesOfFreedomMap = new map<unsigned, DegreeOfFreedom*>();
        totalDegreesOfFreedomMapInverse = new map<DegreeOfFreedom*, unsigned>();
        
        _initiateInternalNodeDOFs(mesh, degreesOfFreedom);
        if (domainBoundaryConditions->varyWithNode())
            _initiateBoundaryNodeDOFWithNonHomogenousBC(mesh, degreesOfFreedom, domainBoundaryConditions);
        else
            _initiateBoundaryNodeDOFWithHomogenousBC(mesh, degreesOfFreedom, domainBoundaryConditions);
        _removeDuplicatesAndDelete(mesh);
        _assignDOFIDs();
        _reconstructTotalDOFList();
        _assignDOFIDsToNodes(mesh);
        _createTotalDOFMap(mesh);
        _listPtrToVectorPtr(freeDegreesOfFreedom, _freeDegreesOfFreedomList);
        _listPtrToVectorPtr(boundedDegreesOfFreedom, _boundedDegreesOfFreedomList);
        for (auto & fluxDOF : *_fluxDegreesOfFreedomList){
            fluxDegreesOfFreedom->push_back(fluxDOF);
        }
        _listPtrToVectorPtr(totalDegreesOfFreedom, _totalDegreesOfFreedomList);
        
        delete _freeDegreesOfFreedomList;
        delete _boundedDegreesOfFreedomList;
        delete _fluxDegreesOfFreedomList;
        delete _totalDegreesOfFreedomList;        
        //printDOFList(_totalDegreesOfFreedomList);
    }
    
    void DOFInitializer::_initiateInternalNodeDOFs(Mesh *mesh, Field_DOFType *degreesOfFreedom) const {
        //March through the mesh nodes
        for (auto & internalNode : *mesh->internalNodesVector){
            for (auto & DOFType : *degreesOfFreedom->DegreesOfFreedom){
                auto dof = new DegreeOfFreedom(DOFType, internalNode->id.global, false);
                _freeDegreesOfFreedomList->push_back(dof);
                _totalDegreesOfFreedomList->push_back(dof);
            }            
        }
    }
    
    void DOFInitializer::_initiateBoundaryNodeDOFWithHomogenousBC(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                                  DomainBoundaryConditions *domainBoundaryConditions) const {
        
        
        for (auto & domainBoundary : *mesh->boundaryNodes){
            auto position = domainBoundary.first;
            auto nodesAtPosition = domainBoundary.second;
            auto bcAtPosition = domainBoundaryConditions-> getBoundaryConditionAtPosition(position);
            //March through the nodes at the Position  that is the key of the domainBoundary map
            for (auto & node : *nodesAtPosition){
                auto dofValue = -1.0;
                for (auto& dofType : *problemDOFTypes->DegreesOfFreedom){
                    dofValue = bcAtPosition->scalarValueOfDOFAt((*dofType));
                    switch (bcAtPosition->type()){
                        case Dirichlet:{
                            auto dirichletDOF = new DegreeOfFreedom(dofType, dofValue, node->id.global, true);
                            _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            _totalDegreesOfFreedomList->push_back(dirichletDOF);
                            break;
                        }
                        case Neumann:{
                            auto neumannDOF = new DegreeOfFreedom(dofType, node->id.global, false);
                            _fluxDegreesOfFreedomList->push_back(tuple<DegreeOfFreedom*, double>(neumannDOF, dofValue));
                            _freeDegreesOfFreedomList->push_back(neumannDOF);
                            _totalDegreesOfFreedomList->push_back(neumannDOF);
                            break;
                        }
                        default:
                            throw std::invalid_argument("Boundary condition type not recognized");
                    }
                }
            }
        }
    }

    void DOFInitializer::_initiateBoundaryNodeDOFWithNonHomogenousBC(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                                  DomainBoundaryConditions *domainBoundaryConditions) const {


        for (auto & domainBoundary : *mesh->boundaryNodes){
            auto position = domainBoundary.first;
            auto nodesAtPosition = domainBoundary.second;
            //March through the nodes at the Position  that is the key of the domainBoundary map
            for (auto & node : *nodesAtPosition){
                auto bcAtPosition = domainBoundaryConditions-> getBoundaryConditionAtPositionAndNode(position,  *node->id.global);
                auto dofValue = -1.0;
                for (auto& dofType : *problemDOFTypes->DegreesOfFreedom){
                    dofValue = bcAtPosition->scalarValueOfDOFAt((*dofType));
                    switch (bcAtPosition->type()){
                        case Dirichlet:{
                            auto dirichletDOF = new DegreeOfFreedom(dofType, dofValue, node->id.global, true);
                            _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            _totalDegreesOfFreedomList->push_back(dirichletDOF);
                            break;
                        }
                        case Neumann:{
                            auto neumannDOF = new DegreeOfFreedom(dofType, node->id.global, false);
                            _fluxDegreesOfFreedomList->push_back(tuple<DegreeOfFreedom*, double>(neumannDOF, dofValue));
                            _freeDegreesOfFreedomList->push_back(neumannDOF);
                            _totalDegreesOfFreedomList->push_back(neumannDOF);
                            break;
                        }
                        default:
                            throw std::invalid_argument("Boundary condition type not recognized");
                    }
                }
            }
        }
    }
    
    void DOFInitializer::_removeDuplicatesAndDelete(Mesh *mesh) const {

        //Sort the list of total DOFs
        _totalDegreesOfFreedomList->sort([](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            return (*a->parentNode) < (*b->parentNode);
        });
        _boundedDegreesOfFreedomList->sort([&mesh](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            return (a->type()) < (b->type());
        });
        _fluxDegreesOfFreedomList->sort([](tuple<DegreeOfFreedom *, double> a, tuple<DegreeOfFreedom *, double> b) {
            return (*get<0>(a)->parentNode) < (*get<0>(b)->parentNode);
        });
        _freeDegreesOfFreedomList->sort([](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            return (*a->parentNode) < (*b->parentNode);
        });



        auto duplicates = vector<DegreeOfFreedom *>();
        //Remove the duplicates from the list of total DOFs. The duplicates are added to the duplicates vector.
        _totalDegreesOfFreedomList->unique([&duplicates](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            auto parentNodeCondition = (*a->parentNode) == (*b->parentNode);
            auto dofTypeCondition = a->type() == b->type();
            auto constraintTypeCondition = false;
            //auto aDirichletBNeumann = a->id->constraintType() == ConstraintType::Fixed && b->id->constraintType() == ConstraintType::Free;

            constraintTypeCondition = a->id->constraintType() == Fixed && b->id->constraintType() == Free;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition) {
                duplicates.push_back(b);
                return true;
            }
            constraintTypeCondition = a->id->constraintType() == Free && b->id->constraintType() == Fixed;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition) {
                duplicates.push_back(a);
                return true;
            }
            constraintTypeCondition = a->id->constraintType() == Fixed && b->id->constraintType() == Fixed;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition) {
                duplicates.push_back(b);
                return true;
            }
            constraintTypeCondition = a->id->constraintType() == Free && b->id->constraintType() == Free;
            if (parentNodeCondition && dofTypeCondition && constraintTypeCondition) {
                duplicates.push_back(b);
                return true;
            }
            return false;
        });
        //Remove the duplicates of the duplicates vector from the bounded or free and flux DOF lists according to
        // the constraint type their value (fixed or free).
        for (auto &dof : duplicates) {
            if (dof->id->constraintType() == ConstraintType::Fixed) {
                _boundedDegreesOfFreedomList->remove(dof);
            }
            else if (dof->id->constraintType() == ConstraintType::Free) {
                //FluxDOFs list has tuples of DOF and flux value. Delete the DOF from the list only from the first item of the tuple
                for (auto &fluxDOF : *_fluxDegreesOfFreedomList) {
                    if (get<0>(fluxDOF) == dof) {
                        _fluxDegreesOfFreedomList->remove(fluxDOF);
                        break;
                    }
                }
                _freeDegreesOfFreedomList->remove(dof);
            }
            delete dof;
            dof = nullptr;
        }
        _boundedDegreesOfFreedomList->unique([&duplicates](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            if (*a->parentNode == *b->parentNode && a->type() == b->type()) {
                return true;
            }
            return false;
        });
        _boundedDegreesOfFreedomList->sort([&mesh](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            auto aNodeBoundaryID = (*mesh->nodeFromID((*a->parentNode))->id.boundary);
            auto bNodeBoundaryID = (*mesh->nodeFromID((*b->parentNode))->id.boundary);
            return aNodeBoundaryID < bNodeBoundaryID;
        });
        
    }

    void DOFInitializer::_assignDOFIDs() const {
        unsigned dofID = 0;
        for (auto &dof : *_freeDegreesOfFreedomList) {
            (*dof->id->value) = dofID;
            //cout << "FREE DOF ID: " << dofID << " Node: " << (*dof->parentNode)<< endl;
            dofID++;
        }
        //cout <<" "<< endl;
        dofID = 0;
        for (auto &dof : *_boundedDegreesOfFreedomList) {
            (*dof->id->value) = dofID;
            cout << "FIXED DOF ID: " << dofID << " Node: " << (*dof->parentNode)<< endl;
            dofID++;
        }
        for (auto &boundedDof: *_boundedDegreesOfFreedomList) {
            boundedDof->print(true);
        }
    }
    
    void DOFInitializer::_reconstructTotalDOFList() const {
        _freeDegreesOfFreedomList->sort([](DegreeOfFreedom* a, DegreeOfFreedom* b){
            return (*a->id->value) < (*b->id->value);
        });
        _totalDegreesOfFreedomList->clear();
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _freeDegreesOfFreedomList->begin(), _freeDegreesOfFreedomList->end());
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _boundedDegreesOfFreedomList->begin(), _boundedDegreesOfFreedomList->end());
        _totalDegreesOfFreedomList->sort([](DegreeOfFreedom* a, DegreeOfFreedom* b){
            return (*a->parentNode) < (*b->parentNode);
        });
    }
    
    void DOFInitializer::_assignDOFIDsToNodes(Mesh *mesh) const {
        for (auto &dof : *_totalDegreesOfFreedomList) {
            auto node = mesh->nodeFromID(((*dof->parentNode)));
            node->degreesOfFreedom->push_back(dof);
        }
    }
    
    void DOFInitializer::_createTotalDOFMap(Mesh *mesh) const {
        auto dofId = 0;
        for (auto &node : *mesh->totalNodesVector) {
            for (auto &dof : *node->degreesOfFreedom) {
                totalDegreesOfFreedomMap->insert(pair<unsigned, DegreeOfFreedom*>(dofId, dof));
                totalDegreesOfFreedomMapInverse->insert(pair<DegreeOfFreedom*, unsigned>(dof, dofId));
                //cout<< "DOF ID: " << dofId << " Node: " << *totalDegreesOfFreedomMap->at(dofId)->parentNode << endl;
                dofId++;
            }
        }
    }
    
    void DOFInitializer::_listPtrToVectorPtr(vector<DegreeOfFreedom *> *vector, list<DegreeOfFreedom *> *list) {
        for (auto &dof : *list) {
            vector->push_back(dof);
        }
    }
}

        