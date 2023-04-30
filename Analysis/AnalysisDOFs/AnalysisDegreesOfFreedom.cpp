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
        fixedDegreesOfFreedom = dofInitializer.boundedDegreesOfFreedom;
        fluxDegreesOfFreedom = dofInitializer.fluxDegreesOfFreedom;
        totalDegreesOfFreedomMap = dofInitializer.totalDegreesOfFreedomMap;
        totalDegreesOfFreedomMapInverse = dofInitializer.totalDegreesOfFreedomMapInverse;

        //printDOFCount();
    }
    
    AnalysisDegreesOfFreedom::~AnalysisDegreesOfFreedom() {
        _deallocateDegreesOfFreedom();
        delete totalDegreesOfFreedom;
        delete freeDegreesOfFreedom;
        delete fixedDegreesOfFreedom;
        delete fluxDegreesOfFreedom;
        totalDegreesOfFreedom = nullptr;
        freeDegreesOfFreedom = nullptr;
        fixedDegreesOfFreedom = nullptr;
        fluxDegreesOfFreedom = nullptr;
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
            delete dof;
        }
    }
} // NumericalAnalysis