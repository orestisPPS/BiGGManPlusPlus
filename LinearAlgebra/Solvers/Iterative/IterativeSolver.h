//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_ITERATIVESOLVER_H
#define UNTITLED_ITERATIVESOLVER_H

#include "../Solver.h"
#include "../../AnalysisLinearSystemInitializer.h"
#include "../../Norms/VectorNorm.h"

namespace LinearAlgebra {

    class IterativeSolver : public Solver {
    public:
        explicit IterativeSolver(VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4);
        
        ~IterativeSolver();

        void setTolerance(double tolerance);
        
        const double& getTolerance() const;
        
        void setMaxIterations(unsigned maxIterations);
        
        const unsigned& getMaxIterations() const;
        
        void setNormType(VectorNormType normType);
        
        const VectorNormType& getNormType() const;
        
        void setLinearSystem(LinearSystem* linearSystem) override;
        
        void setInitialSolution(unique_ptr<vector<double>> initialValue);
        
        void setInitialSolution(double initialValue);
        
    protected:
        
        VectorNormType _normType;
        
        double _tolerance;
        
        unsigned _maxIterations;
        
        unique_ptr<vector<double>> _xInitial;
        
        unique_ptr<vector<double>> _xNew;
        
        unique_ptr<vector<double>> _xOld;
        
        bool _isInitialized;
        
    };
    



} // LinearAlgebra

#endif //UNTITLED_ITERATIVESOLVER_H
