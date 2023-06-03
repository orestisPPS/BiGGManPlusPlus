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
                auto dof = new DegreeOfFreedom(DOFType, internalNode->id.global, false);
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
                            auto dirichletDOF = new DegreeOfFreedom(dofType, dofValue, node->id.global, true);
                            _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            break;
                        }
                        case Neumann:{
                            auto neumannDOF = new DegreeOfFreedom(dofType, node->id.global, false);
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
                auto bcAtPosition = domainBoundaryConditions-> getBoundaryConditionAtPositionAndNode(position,  *node->id.global);
                auto dofValue = -1.0;
                for (auto& dofType : *problemDOFTypes->DegreesOfFreedom){
                    dofValue = bcAtPosition->scalarValueOfDOFAt((*dofType));
                    switch (bcAtPosition->type()){
                        case Dirichlet:{
                            auto dirichletDOF = new DegreeOfFreedom(dofType, dofValue, node->id.global, true);
                            _boundedDegreesOfFreedomList->push_back(dirichletDOF);
                            break;
                        }
                        case Neumann:{
                            auto neumannDOF = new DegreeOfFreedom(dofType, node->id.global, false);
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

    void DOFInitializer::_removeDuplicatesAndDelete(Mesh *mesh) const {
        
        _boundedDegreesOfFreedomList->sort([](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            if (*a->parentNode == *b->parentNode)
                return a->type() < b->type();
            return *a->parentNode < *b->parentNode;
        });
        //THIS IS WORKING
        _boundedDegreesOfFreedomList->unique([](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            return *a->parentNode == *b->parentNode && a->type() == b->type();
   /*         if (*a->parentNode == *b->parentNode && a->type() == b->type()){
                auto meanValue = (a->value() + b->value())/2.0;
                a->setValue(meanValue);
                delete b;
                b = nullptr;
                return true;
            }*/
            return false;
        });
        _fluxDegreesOfFreedomList->sort([](tuple<DegreeOfFreedom *, double> a, tuple<DegreeOfFreedom *, double> b) {
            return (*get<0>(a)->parentNode) < (*get<0>(b)->parentNode);
        });
        _freeDegreesOfFreedomList->sort([](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            return (*a->parentNode) < (*b->parentNode);
        });

/*        _boundedDegreesOfFreedomList->unique([](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            auto parentA = *a->parentNode;
            auto parentB = *b->parentNode;
            auto parentCondition = parentA == parentB;
            auto typeA = a->type();
            auto typeB = b->type();
            auto typeCondition = typeA == typeB;
            if (*a->parentNode == *b->parentNode && a->type() == b->type()) {
                return true;
            }
            return false;
        });*/
/*        _boundedDegreesOfFreedomList->sort([&mesh](DegreeOfFreedom *a, DegreeOfFreedom *b) {
            auto aNodeGlobalID = (*mesh->nodeFromID((*a->parentNode))->id.global);
            auto bNodeGlobalID = (*mesh->nodeFromID((*b->parentNode))->id.global);
            return aNodeGlobalID < bNodeGlobalID;
        });*/
        
/*        for (auto &dof : *_boundedDegreesOfFreedomList) {
            dof->print(true);
        }*/

    }
    
    void DOFInitializer::_createTotalDOFList(Mesh* mesh) const {
        _freeDegreesOfFreedomList->sort([](DegreeOfFreedom* a, DegreeOfFreedom* b){
            return (*a->parentNode) < (*b->parentNode);
        });
        _boundedDegreesOfFreedomList->sort([](DegreeOfFreedom* a, DegreeOfFreedom* b){
            return (*a->parentNode) < (*b->parentNode);
        });
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _freeDegreesOfFreedomList->begin(), _freeDegreesOfFreedomList->end());
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _boundedDegreesOfFreedomList->begin(), _boundedDegreesOfFreedomList->end());
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
            //dof->print(true);
            dofID++;
        }

    }
    void DOFInitializer::_assignDOFToNodes(Mesh *mesh) const {
        for (auto &dof : *_totalDegreesOfFreedomList) {
            auto node = mesh->nodeFromID(((*dof->parentNode)));
            node->degreesOfFreedom->push_back(dof);
            //dof->print(false);
        }
    }

    void DOFInitializer::_createTotalDOFDataStructures(Mesh *mesh) const {
        auto dofId = 0;
        for (auto &dof : *_totalDegreesOfFreedomList) {
            dof->id->globalValue = new unsigned int (dofId);
            totalDegreesOfFreedomMap->insert(pair<int, DegreeOfFreedom *>(dofId, dof));
            totalDegreesOfFreedomMapInverse->insert(pair<DegreeOfFreedom *, int>(dof, dofId));
            dofId++;
        }
/*        unsigned dofID = 0;
        for (auto& node : *mesh->totalNodesVector){
            for (auto& dof : *node->degreesOfFreedom){
                dof->id->globalValue = new unsigned int (dofID);
                totalDegreesOfFreedomMap->insert(pair<int, DegreeOfFreedom *>(dofID, dof));
                totalDegreesOfFreedomMapInverse->insert(pair<DegreeOfFreedom *, int>(dof, dofID));
                dofID++;
            }
        }*/
    }

    void DOFInitializer::_listPtrToVectorPtr(vector<DegreeOfFreedom *> *vector, list<DegreeOfFreedom *> *list) {
        for (auto &dof : *list) {
            vector->push_back(dof);
        }
    }
}

        