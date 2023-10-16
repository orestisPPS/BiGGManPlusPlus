//
// Created by hal9000 on 10/6/23.
//

#include <stdexcept>
#include "InitialConditions.h"

namespace MathematicalEntities {
    
    InitialConditions::InitialConditions(double zeroOrderInitialCondition) :
            _constantZeroOrderIC(zeroOrderInitialCondition), _constantFirstOrderIC(0),
            _varyingZeroOrderICs(nullptr), _varyingFirstOrderICs(nullptr) {
    }
    
    InitialConditions::InitialConditions(double zeroOrderInitialCondition, double firstOrderInitialCondition) :
            _constantZeroOrderIC(zeroOrderInitialCondition), _constantFirstOrderIC(firstOrderInitialCondition),
            _varyingZeroOrderICs(nullptr), _varyingFirstOrderICs(nullptr) {
    }
    
    InitialConditions::InitialConditions(unique_ptr<unordered_map<unsigned, double>> zeroOrderInitialConditions) :
            _constantZeroOrderIC(0), _constantFirstOrderIC(0),
            _varyingZeroOrderICs(std::move(zeroOrderInitialConditions)), _varyingFirstOrderICs(nullptr) {
    }
    
    InitialConditions::InitialConditions(unique_ptr<unordered_map<unsigned, double>> firstOrderInitialConditions,
                                         unique_ptr<unordered_map<unsigned, double>> secondOrderInitialConditions) :
            _constantZeroOrderIC(0), _constantFirstOrderIC(0),
            _varyingZeroOrderICs(std::move(firstOrderInitialConditions)),
            _varyingFirstOrderICs(std::move(secondOrderInitialConditions)) {
    }
    
    InitialConditions::~InitialConditions() {
        _varyingFirstOrderICs.reset();
        _varyingZeroOrderICs.reset();
    }
    
    void InitialConditions::setConstantInitialConditions(double zeroOrderInitialCondition) {
        _constantFirstOrderIC = zeroOrderInitialCondition;
    }
    
    void InitialConditions::setConstantInitialConditions(double zeroOrderInitialCondition, double firstOrderInitialConditions) {
        _constantZeroOrderIC = zeroOrderInitialCondition;
        _constantFirstOrderIC = firstOrderInitialConditions;
    }
    
    void InitialConditions::setVaryingInitialConditions(unique_ptr<unordered_map<unsigned, double>> zeroOrderInitialCondition) {
        _varyingZeroOrderICs = std::move(zeroOrderInitialCondition);
    }
    
    void InitialConditions::setVaryingInitialConditions(unique_ptr<unordered_map<unsigned, double>> zeroOrderInitialCondition,
                                                        unique_ptr<unordered_map<unsigned, double>> firstOrderInitialConditions) {
        _varyingZeroOrderICs = std::move(zeroOrderInitialCondition);
        _varyingFirstOrderICs = std::move(firstOrderInitialConditions);
    }
    
    void InitialConditions::setVaryingInitialConditions(unsigned id, double zeroOrderInitialCondition) {
        if (_varyingZeroOrderICs == nullptr) {
            _varyingZeroOrderICs = make_unique<unordered_map<unsigned, double>>();
        }
        _varyingZeroOrderICs->insert({id, zeroOrderInitialCondition});
    }
    
    void InitialConditions::setVaryingInitialConditions(unsigned id, double zeroOrderInitialCondition,
                                                                     double firstOrderInitialConditions) {
        if (_varyingZeroOrderICs == nullptr) {
            _varyingZeroOrderICs = make_unique<unordered_map<unsigned, double>>();
        }
        if (_varyingFirstOrderICs == nullptr) {
            _varyingFirstOrderICs = make_unique<unordered_map<unsigned, double>>();
        }
        _varyingZeroOrderICs->insert({id, zeroOrderInitialCondition});
        _varyingFirstOrderICs->insert({id, firstOrderInitialConditions});
    }
    
    double InitialConditions::getInitialCondition(unsigned derivativeOrder) const {
        if (derivativeOrder == 0) {
            if (_varyingZeroOrderICs != nullptr) {
                throw invalid_argument("The initial conditions for zero derivative are spatially varying");
            }
            return _constantZeroOrderIC;
        }
        else if (derivativeOrder == 1) {
            if (_varyingFirstOrderICs != nullptr) {
                throw invalid_argument("The initial conditions for first derivative are spatially varying");
            }
            return _constantFirstOrderIC;
        }
        else {
            throw invalid_argument("Initial conditions for derivatives of order higher than 1 are not defined");
        }
    }
    
    double InitialConditions::getInitialCondition(unsigned derivativeOrder, unsigned id) {
        if (derivativeOrder > 1) {
            throw invalid_argument("Initial conditions for derivatives of order higher than 1 are not defined");
        }
        if (derivativeOrder == 0) {
            if (_varyingZeroOrderICs == nullptr) {
                throw invalid_argument("The initial conditions for zero derivative are constant");
            }
            return _varyingZeroOrderICs->at(id);
        }
        else if (derivativeOrder == 1) {
            if (_varyingFirstOrderICs == nullptr) {
                throw invalid_argument("The initial conditions for first derivative are constant");
            }
            return _varyingFirstOrderICs->at(id);
        }
    }
    
    void InitialConditions::deallocate() {
        _varyingFirstOrderICs.reset();
        _varyingZeroOrderICs.reset();
    }

} // MathematicalEntities