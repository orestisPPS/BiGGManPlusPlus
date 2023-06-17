//
// Created by hal9000 on 3/12/23.
//

#include "AnalysisDegreesOfFreedom.h"

#include <utility>

namespace NumericalAnalysis {
    
    AnalysisDegreesOfFreedom::AnalysisDegreesOfFreedom(shared_ptr<Mesh> mesh, shared_ptr<DomainBoundaryConditions>domainBoundaryConditions,
                                                       Field_DOFType* degreesOfFreedom) {
        auto dofInitializer = DOFInitializer(mesh, domainBoundaryConditions, degreesOfFreedom);
        totalDegreesOfFreedom = dofInitializer.totalDegreesOfFreedom;
        freeDegreesOfFreedom = dofInitializer.freeDegreesOfFreedom;
        fixedDegreesOfFreedom = dofInitializer.fixedDegreesOfFreedom;
        fluxDegreesOfFreedom = dofInitializer.fluxDegreesOfFreedom;
        totalDegreesOfFreedomMap = dofInitializer.totalDegreesOfFreedomMap;
        totalDegreesOfFreedomMapInverse = dofInitializer.totalDegreesOfFreedomMapInverse;
        numberOfFreeDOF = make_shared<unsigned>(freeDegreesOfFreedom->size());
        numberOfFixedDOF = make_shared<unsigned>(fixedDegreesOfFreedom->size());
        numberOfDOF = make_shared<unsigned>(totalDegreesOfFreedom->size());
        //printDOFCount();
    }
    
    AnalysisDegreesOfFreedom::~AnalysisDegreesOfFreedom() {
        _deallocateDegreesOfFreedom();
    }
    

    void AnalysisDegreesOfFreedom::printDOFCount() const {
        cout << "Degrees of Freedom Initiated" << endl;
        cout << "Total DOFs: " << totalDegreesOfFreedom->size() << endl;
        cout << "Free DOFs: " << freeDegreesOfFreedom->size() << endl;
        cout << "Bounded DOFs: " << fixedDegreesOfFreedom->size() << endl;
        cout << "Flux DOFs: " << fluxDegreesOfFreedom->size() << endl;
    }
    
    void AnalysisDegreesOfFreedom::_deallocateDegreesOfFreedom() const {
        for (auto dof : *totalDegreesOfFreedom) {
            if(dof != nullptr){
                delete dof;
                dof = nullptr;
            }
        }
    }
} // NumericalAnalysis