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
        _createTotalDOFList(mesh);
        _assignDOFIDs();
        _assignDOFToNodes(mesh);
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

    void DOFInitializer::_initiateInternalNodeDOFs(Mesh *mesh, Field_DOFType *degreesOfFreedom) const {
        //March through the mesh nodes
        for (auto & internalNode : *mesh->internalNodesVector){
            for (auto & DOFType : *degreesOfFreedom->DegreesOfFreedom){
                auto dof = new DegreeOfFreedom(DOFType, *internalNode->id.global, false);
                _freeDegreesOfFreedomList->push_back(dof);
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
                            auto dirichletDOF = new DegreeOfFreedom(dofType, *node->id.global, true, dofValue);
                            _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            break;
                        }
                        case Neumann:{
                            auto neumannDOF = new DegreeOfFreedom(dofType, *node->id.global, false);
                            _fluxDegreesOfFreedomList->push_back(tuple<DegreeOfFreedom*, double>(neumannDOF, dofValue));
                            _freeDegreesOfFreedomList->push_back(neumannDOF);
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
                node->printNode();

                auto bcAtPosition = domainBoundaryConditions-> getBoundaryConditionAtPositionAndNode(position,  *node->id.global);
                auto dofValue = -1.0;
                for (auto& dofType : *problemDOFTypes->DegreesOfFreedom){
                    dofValue = bcAtPosition->scalarValueOfDOFAt((*dofType));
                    cout <<"dof value: " << dofValue << endl;
                    switch (bcAtPosition->type()){
                        case Dirichlet:{
                            auto dirichletDOF = new DegreeOfFreedom(dofType, *node->id.global, true, dofValue);
                            _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            break;
                        }
                        case Neumann:{
                            auto neumannDOF = new DegreeOfFreedom(dofType, *node->id.global, false);
                            _fluxDegreesOfFreedomList->push_back(tuple<DegreeOfFreedom*, double>(neumannDOF, dofValue));
                            _freeDegreesOfFreedomList->push_back(neumannDOF);
                            break;
                        }
                        default:
                            throw std::invalid_argument("Boundary condition type not recognized");
                    }
                }
            }
        }
    }

    void DOFInitializer::_removeDuplicatesAndDelete(Mesh* mesh) const {
        _boundedDegreesOfFreedomList->sort([](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            if (a->parentNode() == b->parentNode())
                return a->type() < b->type();
            return a->parentNode() < b->parentNode();
        });

        _boundedDegreesOfFreedomList->unique([](DegreeOfFreedom* a, DegreeOfFreedom* b) {
            if ( a->parentNode() == b->parentNode() && a->type() == b->type()){
/*                auto median = (a->value() + b->value())/2.0;
                a->setValue(median);
                b->setValue(median);*/
                return true;
            }
            return false;
        });
    }

    void DOFInitializer::_createTotalDOFList(Mesh* mesh) const {
        _freeDegreesOfFreedomList->sort([](const DegreeOfFreedom* a, const DegreeOfFreedom* b) {
            return a->parentNode() < b->parentNode();
        });

        _boundedDegreesOfFreedomList->sort([](const DegreeOfFreedom* a, const DegreeOfFreedom* b) {
            return a->parentNode() < b->parentNode();
        });

        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _freeDegreesOfFreedomList->begin(), _freeDegreesOfFreedomList->end());
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _boundedDegreesOfFreedomList->begin(), _boundedDegreesOfFreedomList->end());

        _fluxDegreesOfFreedomList->sort([](const tuple<DegreeOfFreedom*, double>& a, const tuple<DegreeOfFreedom*, double>& b) {
            return get<0>(a)->parentNode() < get<0>(b)->parentNode();
        });
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
    void DOFInitializer::_assignDOFToNodes(Mesh *mesh) const {
        for (auto &dof : *_totalDegreesOfFreedomList) {
            auto node = mesh->nodeFromID(dof->parentNode());
            node->degreesOfFreedom->push_back(dof);
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

        