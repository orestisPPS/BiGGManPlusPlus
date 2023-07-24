//
// Created by hal9000 on 6/14/23.
//

#ifndef UNTITLED_SORSOLVER_H
#define UNTITLED_SORSOLVER_H

#include "StationaryIterative.h"

class SORSolver : public StationaryIterative {
    
public:
    SORSolver(double relaxationParameter, ParallelizationMethod parallelizationMethod, VectorNormType normType, double tolerance = 1E-9, unsigned maxIterations = 1E4, bool throwExceptionOnMaxFailure = true);

protected:
    
    void _threadJob(unsigned start, unsigned end) override;
    
    //TODO create a function that allows to change the relaxation parameter mid calculation
    
    //Relaxation parameter
    double _relaxationParameter;

};


#endif //UNTITLED_SORSOLVER_H
