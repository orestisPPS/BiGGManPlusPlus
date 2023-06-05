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
        
        _createTotalDOFList(mesh);
        _assignDOFIDs();
        _createTotalDOFDataStructures(mesh);
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

    void DOFInitializer::_initiateInternalNodeDOFs(Mesh *mesh, Field_DOFType *degreesOfFreedom){
        //March through the mesh nodes
        for (auto & internalNode : *mesh->internalNodesVector){
            for (auto & DOFType : *degreesOfFreedom->DegreesOfFreedom){
                auto dof = new DegreeOfFreedom(DOFType, *internalNode->id.global, false);
                internalNode->degreesOfFreedom->push_back(dof);
                _freeDegreesOfFreedomList->push_back(dof);
            }
        }
    }

    void DOFInitializer::_initiateBoundaryNodeDOFWithHomogenousBC(Mesh *mesh, Field_DOFType *problemDOFTypes,
                                                                  DomainBoundaryConditions *domainBoundaryConditions) {
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
                            auto isDuplicate = false;
                            for (auto &dof : *node->degreesOfFreedom){
                                if (dof->type() == *dofType && dof->constraintType() == Fixed){
                                    auto median = (dof->value() + dofValue)/2.0;
                                    dof->setValue(median);
                                    isDuplicate = true;
                                }
                            }
                            if (!isDuplicate){
                                auto dirichletDOF = new DegreeOfFreedom(dofType, *node->id.global, true, dofValue);
                                node->degreesOfFreedom->push_back(dirichletDOF);
                                _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            }
                            break;
                        }
                        case Neumann:{
                            auto median = 0.0;
                            auto isDuplicate = false;
                            for (auto &dof : *node->degreesOfFreedom){
                                if (dof->type() == *dofType && dof->constraintType() == Free){
                                    //Search if _fluxDegreesOfFreedomList already contains a DOF of the same type
                                    for (auto &fluxDOF : *_fluxDegreesOfFreedomList){
                                        if (get<0>(fluxDOF)->type() == *dofType && get<0>(fluxDOF)->parentNode()== *node->id.global){
                                            isDuplicate = true;
                                            median = (get<1>(fluxDOF) + dofValue)/2.0;
                                            get<1>(fluxDOF) = median;
                                        }
                                    }
                                }
                            }
                            if (!isDuplicate){
                                auto neumannDOF = new DegreeOfFreedom(dofType, *node->id.global, false);
                                node->degreesOfFreedom->push_back(neumannDOF);
                                _fluxDegreesOfFreedomList->push_back(tuple<DegreeOfFreedom*, double>(neumannDOF, dofValue));
                                _freeDegreesOfFreedomList->push_back(neumannDOF);
                            }
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
                                                                     DomainBoundaryConditions *domainBoundaryConditions) {
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
                            auto isDuplicate = false;
                            for (auto &dof : *node->degreesOfFreedom){
                                if (dof->type() == *dofType && dof->constraintType() == Fixed){
                                    auto median = (dof->value() + dofValue)/2.0;
                                    dof->setValue(median);
                                    isDuplicate = true;
                                }
                            }
                            if (!isDuplicate){
                                auto dirichletDOF = new DegreeOfFreedom(dofType, *node->id.global, true, dofValue);
                                node->degreesOfFreedom->push_back(dirichletDOF);
                                _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            }
                            break;
                        }
                        case Neumann:{
                            auto median = 0.0;
                            auto isDuplicate = false;
                            for (auto &dof : *node->degreesOfFreedom){
                                if (dof->type() == *dofType && dof->constraintType() == Free){
                                    //Search if _fluxDegreesOfFreedomList already contains a DOF of the same type
                                    for (auto &fluxDOF : *_fluxDegreesOfFreedomList){
                                        if (get<0>(fluxDOF)->type() == *dofType && get<0>(fluxDOF)->parentNode()== *node->id.global){
                                            isDuplicate = true;
                                            median = (get<1>(fluxDOF) + dofValue)/2.0;
                                            get<1>(fluxDOF) = median;
                                        }
                                    }
                                }
                            }
                            if (!isDuplicate){
                                auto neumannDOF = new DegreeOfFreedom(dofType, *node->id.global, false);
                                node->degreesOfFreedom->push_back(neumannDOF);
                                _fluxDegreesOfFreedomList->push_back(tuple<DegreeOfFreedom*, double>(neumannDOF, dofValue));
                                _freeDegreesOfFreedomList->push_back(neumannDOF);
                            }
                            break;
                        }
                        default:
                            throw std::invalid_argument("Boundary condition type not recognized");
                    }
                }
            }
        }
    }
    

    void DOFInitializer::_createTotalDOFList(Mesh* mesh) const {
        _freeDegreesOfFreedomList->sort([](const DegreeOfFreedom* a, const DegreeOfFreedom* b) {
            return a->parentNode() < b->parentNode();
        });

        _boundedDegreesOfFreedomList->sort([](const DegreeOfFreedom* a, const DegreeOfFreedom* b) {
            return a->parentNode() < b->parentNode();
        });
        _fluxDegreesOfFreedomList->sort([](const tuple<DegreeOfFreedom*, double>& a, const tuple<DegreeOfFreedom*, double>& b) {
            return get<0>(a)->parentNode() < get<0>(b)->parentNode();
        });
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _freeDegreesOfFreedomList->begin(), _freeDegreesOfFreedomList->end());
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _boundedDegreesOfFreedomList->begin(), _boundedDegreesOfFreedomList->end());
    }
    
    void DOFInitializer::_assignDOFIDs() const {
        unsigned freeDofID = 0;
        for (auto &dof : *_freeDegreesOfFreedomList) {
            dof->setID(freeDofID);
            freeDofID++;
        }
        auto fixedDofID = 0;
        for (auto &dof : *_boundedDegreesOfFreedomList) {
            dof->setID(fixedDofID);
            fixedDofID++;
        }

    }

    void DOFInitializer::_createTotalDOFDataStructures(Mesh *mesh) const {
        auto dofId = 0;
        for (auto &dof : *_totalDegreesOfFreedomList) {
            totalDegreesOfFreedomMap->insert(pair<int, DegreeOfFreedom *>(dofId, dof));
            totalDegreesOfFreedomMapInverse->insert(pair<DegreeOfFreedom *, int>(dof, dofId));
            dofId++;
        }
    }

    void DOFInitializer::_listPtrToVectorPtr(vector<DegreeOfFreedom *> *vector, list<DegreeOfFreedom *> *list) {
        for (auto &dof : *list) {
            vector->push_back(dof);
        }
    }
}

        