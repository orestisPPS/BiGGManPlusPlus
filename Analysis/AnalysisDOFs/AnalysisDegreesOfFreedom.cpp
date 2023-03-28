//
// Created by hal9000 on 3/12/23.
//

#include "AnalysisDegreesOfFreedom.h"

namespace NumericalAnalysis {
    
    AnalysisDegreesOfFreedom::AnalysisDegreesOfFreedom(Mesh *mesh, DomainBoundaryConditions *domainBoundaryConditions,
                                                       Field_DOFType* degreesOfFreedom) {
        auto dofInitializer = DOFInitializer(mesh, domainBoundaryConditions, degreesOfFreedom);
        totalDegreesOfFreedom = dofInitializer.totalDegreesOfFreedom;
        freeDegreesOfFreedom = dofInitializer.freeDegreesOfFreedom;
        boundedDegreesOfFreedom = dofInitializer.boundedDegreesOfFreedom;
        fluxDegreesOfFreedom = dofInitializer.fluxDegreesOfFreedom;
        
        printDOFCount();
    }
    
    AnalysisDegreesOfFreedom::~AnalysisDegreesOfFreedom() {
        _deallocateDegreesOfFreedom();
        delete totalDegreesOfFreedom;
        delete freeDegreesOfFreedom;
        delete boundedDegreesOfFreedom;
        delete fluxDegreesOfFreedom;
        totalDegreesOfFreedom = nullptr;
        freeDegreesOfFreedom = nullptr;
        boundedDegreesOfFreedom = nullptr;
        fluxDegreesOfFreedom = nullptr;
    }
    
/*    map<unsigned*, vector<DegreeOfFreedom*>*> AnalysisDegreesOfFreedom::_createNodeDofMap(Mesh *mesh, Field_DOFType* degreesOfFreedom) {
        map<unsigned*, vector<DegreeOfFreedom*>*> nodeDofMap;
        for (auto &dof : *totalDegreesOfFreedom){
            auto node = dof->parentNode;
            if (nodeDofMap.find(node) == nodeDofMap.end()){
                nodeDofMap[node] = mesh->nodeFromID(*node)->degreesOfFreedom;
            }
            nodeDofMap[node] = new vector<DegreeOfFreedom*>();
        }
    }*/

    void AnalysisDegreesOfFreedom::printDOFCount() const {
        cout << "Degrees of Freedom Initiated" << endl;
        cout << "Total DOFs: " << totalDegreesOfFreedom->size() << endl;
        cout << "Free DOFs: " << freeDegreesOfFreedom->size() << endl;
        cout << "Bounded DOFs: " << boundedDegreesOfFreedom->size() << endl;
        cout << "Flux DOFs: " << fluxDegreesOfFreedom->size() << endl;
    }
    
    void AnalysisDegreesOfFreedom::_deallocateDegreesOfFreedom() const {
        for (auto dof : *totalDegreesOfFreedom) {
            delete dof;
        }
    }
} // NumericalAnalysis