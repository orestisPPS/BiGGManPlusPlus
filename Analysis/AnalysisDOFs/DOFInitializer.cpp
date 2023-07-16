//
// Created by hal9000 on 2/18/23.
//

#include <algorithm>
#include "DOFInitializer.h"

namespace NumericalAnalysis {

    DOFInitializer::DOFInitializer(const shared_ptr<Mesh>& mesh, const shared_ptr<DomainBoundaryConditions>& domainBoundaryConditions, Field_DOFType* degreesOfFreedom) {

        _totalDegreesOfFreedomList = make_shared<list<DegreeOfFreedom*>>();
        _freeDegreesOfFreedomList = make_shared<list<DegreeOfFreedom*>>();
        _fixedDegreesOfFreedomList = make_shared<list<DegreeOfFreedom*>>();
        _internalDegreesOfFreedomList = make_shared<list<DegreeOfFreedom*>>();

        totalDegreesOfFreedom = make_shared<vector<DegreeOfFreedom*>>();
        freeDegreesOfFreedom = make_shared<vector<DegreeOfFreedom*>>();
        fixedDegreesOfFreedom = make_shared<vector<DegreeOfFreedom*>>();
        internalDegreesOfFreedom = make_shared<vector<DegreeOfFreedom*>>();
        fluxDegreesOfFreedom = make_shared<map<DegreeOfFreedom*, double>>();

        totalDegreesOfFreedomMap = make_shared<map<unsigned, DegreeOfFreedom*>>();
        totalDegreesOfFreedomMapInverse = make_shared<map<DegreeOfFreedom*, unsigned>>();
        
        _initiateInternalNodeDOFs(mesh, degreesOfFreedom);
        
        if (domainBoundaryConditions->varyWithNode())
            _initiateBoundaryNodeDOFWithNonHomogenousBC(mesh, degreesOfFreedom, domainBoundaryConditions);
        else
            _initiateBoundaryNodeDOFWithHomogenousBC(mesh, degreesOfFreedom, domainBoundaryConditions);
        
        _createTotalDOFList(mesh);
        _assignDOFIDs();
        _createTotalDOFDataStructures(mesh);
        _listPtrToVectorPtr(_freeDegreesOfFreedomList, freeDegreesOfFreedom);
        _listPtrToVectorPtr(_fixedDegreesOfFreedomList, fixedDegreesOfFreedom);
        _listPtrToVectorPtr(_totalDegreesOfFreedomList, totalDegreesOfFreedom);
        _listPtrToVectorPtr(_internalDegreesOfFreedomList, internalDegreesOfFreedom);
    }

    void DOFInitializer::_initiateInternalNodeDOFs(const shared_ptr<Mesh>& mesh, Field_DOFType *degreesOfFreedom){
        auto internalNodesVector = mesh->getInternalNodesVector();
        //March through the mesh nodes
        for (auto & internalNode : *internalNodesVector){
            for (auto & DOFType : *degreesOfFreedom->DegreesOfFreedom){
                auto dof = new DegreeOfFreedom(DOFType, *internalNode->id.global, false);
                internalNode->degreesOfFreedom->push_back(dof);
                _internalDegreesOfFreedomList->push_back(dof);
                _freeDegreesOfFreedomList->push_back(dof);
            }
        }
    }

    void DOFInitializer::_initiateBoundaryNodeDOFWithHomogenousBC(const shared_ptr<Mesh> &mesh, Field_DOFType *problemDOFTypes,
                                                                  const shared_ptr<DomainBoundaryConditions> &domainBoundaryConditions) {
        for (auto & domainBoundary : *mesh->boundaryNodes){
            auto position = domainBoundary.first;
            auto nodesAtPosition = domainBoundary.second;
            auto bcAtPosition = domainBoundaryConditions-> getBoundaryConditionAtPosition(position);
            //March through the nodes at the Position  that is the key of the domainBoundary map
            for (auto & node : *nodesAtPosition){
                auto dofValue = -1.0;
                for (auto& dofType : *problemDOFTypes->DegreesOfFreedom){
                    dofValue = bcAtPosition->getBoundaryConditionValue((*dofType));
                    switch (bcAtPosition->type()){
                        case Dirichlet:{
                            auto isDuplicate = false;
                            for (auto &dof : *node->degreesOfFreedom){
                                if (dof->type() == *dofType && dof->constraintType() == Fixed)
                                    break;
                                if (dof->type() == *dofType && dof->constraintType() == Fixed){
                                    auto median = (dof->value() + dofValue)/2.0;
                                    dof->setValue(median);
                                    isDuplicate = true;
                                }
                            }
                            if (!isDuplicate){
                                auto dirichletDOF = new DegreeOfFreedom(dofType, *node->id.global, true, dofValue);
                                node->degreesOfFreedom->push_back(dirichletDOF);
                                _fixedDegreesOfFreedomList->push_back(dirichletDOF);
                            }
                            break;
                        }
                        case Neumann:{
                            auto median = 0.0;
                            auto isDuplicate = false;
                            for (auto &dof : *node->degreesOfFreedom){
                                if (dof->type() == *dofType && dof->constraintType() == Fixed)
                                    isDuplicate = true;
                                else if (dof->type() == *dofType && dof->constraintType() == Free){
                                    //Search if fluxDegreesOfFreedom already contains a DOF of the same type
                                    for (auto &fluxDOF : *fluxDegreesOfFreedom){
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
                                fluxDegreesOfFreedom->insert(pair<DegreeOfFreedom*, double>(neumannDOF, dofValue));
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

    void DOFInitializer::_initiateBoundaryNodeDOFWithNonHomogenousBC(const shared_ptr<Mesh>& mesh, Field_DOFType *problemDOFTypes,
                                                                     const shared_ptr<DomainBoundaryConditions>&domainBoundaryConditions) {
        for (auto & domainBoundary : *mesh->boundaryNodes){
            auto position = domainBoundary.first;
            auto nodesAtPosition = domainBoundary.second;
            //March through the nodes at the Position  that is the key of the domainBoundary map
            for (auto & node : *nodesAtPosition){

                auto bcAtPosition = domainBoundaryConditions-> getBoundaryConditionAtPositionAndNode(position,  *node->id.global);
                auto dofValue = -1.0;
                for (auto& dofType : *problemDOFTypes->DegreesOfFreedom){
                    dofValue = bcAtPosition->getBoundaryConditionValue((*dofType));
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
                                _fixedDegreesOfFreedomList->push_back(dirichletDOF);
                            }
                            break;
                        }
                        case Neumann:{
                            auto median = 0.0;
                            auto isDuplicate = false;
                            for (auto &dof : *node->degreesOfFreedom){
                                if (dof->type() == *dofType && dof->constraintType() == Fixed)
                                    break;
                                if (dof->type() == *dofType && dof->constraintType() == Free){
                                    //Search if fluxDegreesOfFreedom already contains a DOF of the same type
                                    for (auto &fluxDOF : *fluxDegreesOfFreedom){
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
                                fluxDegreesOfFreedom->insert(pair<DegreeOfFreedom*, double>(neumannDOF, dofValue));
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
    

    void DOFInitializer::_createTotalDOFList(const shared_ptr<Mesh>& mesh) const {

        auto fluxDOFList = make_unique<list<DegreeOfFreedom*>>();
        for (auto & fluxDOF : *fluxDegreesOfFreedom){
            fluxDOFList->push_back(get<0>(fluxDOF));
        }

        fluxDOFList->sort([](const DegreeOfFreedom* a, const DegreeOfFreedom* b) {
            return a->parentNode() < b->parentNode();
        });
        
        _freeDegreesOfFreedomList->sort([](const DegreeOfFreedom* a, const DegreeOfFreedom* b) {
            return a->parentNode() < b->parentNode();
        });

        _fixedDegreesOfFreedomList->sort([](const DegreeOfFreedom* a, const DegreeOfFreedom* b) {
            return a->parentNode() < b->parentNode();
        });
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _freeDegreesOfFreedomList->begin(), _freeDegreesOfFreedomList->end());
        _totalDegreesOfFreedomList->insert(_totalDegreesOfFreedomList->end(),
                                           _fixedDegreesOfFreedomList->begin(), _fixedDegreesOfFreedomList->end());
    }
    
    void DOFInitializer::_assignDOFIDs() const {
        unsigned freeDofID = 0;
        for (auto &dof : *_freeDegreesOfFreedomList) {
            dof->setID(freeDofID);
            freeDofID++;
        }
        auto fixedDofID = 0;
        for (auto &dof : *_fixedDegreesOfFreedomList) {
            dof->setID(fixedDofID);
            fixedDofID++;
        }

    }

    void DOFInitializer::_createTotalDOFDataStructures(const shared_ptr<Mesh>& mesh) const {
        auto dofId = 0;
        for (auto &dof : *_totalDegreesOfFreedomList) {
            totalDegreesOfFreedomMap->insert(pair<int, DegreeOfFreedom *>(dofId, dof));
            totalDegreesOfFreedomMapInverse->insert(pair<DegreeOfFreedom *, int>(dof, dofId));
            dofId++;
        }
    }

    void DOFInitializer::_listPtrToVectorPtr(const shared_ptr<list<DegreeOfFreedom *>>& list, const shared_ptr<vector<DegreeOfFreedom *>>& vector) {
        for (auto &dof : *list) {
            vector->push_back(dof);
        }
    }
}

        