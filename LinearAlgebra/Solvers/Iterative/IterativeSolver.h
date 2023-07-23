//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_ITERATIVESOLVER_H
#define UNTITLED_ITERATIVESOLVER_H

#include <thread>
#include "../Solver.h"
#include "../../AnalysisLinearSystemInitializer.h"
#include "../../Norms/VectorNorm.h"

namespace LinearAlgebra {

    class IterativeSolver : public Solver {
    public:
        explicit IterativeSolver(VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4, bool throwExceptionOnMaxFailure = true);
        
        ~IterativeSolver();

        void setTolerance(double tolerance);
        
        const double& getTolerance() const;
        
        void setMaxIterations(unsigned maxIterations);
        
        const unsigned& getMaxIterations() const;
        
        void setNormType(VectorNormType normType);
        
        const VectorNormType& getNormType() const;
        
        void setLinearSystem(shared_ptr<LinearSystem> linearSystem) override;
        
        void setInitialSolution(shared_ptr<vector<double>> initialValue);
        
        void setInitialSolution(double initialValue);
        
        void solve() override;
        
    protected:
        
        
        virtual void _iterativeSolution();
        
        VectorNormType _normType;
        
        double _tolerance;
        
        unsigned _maxIterations;
        
        shared_ptr<vector<double>> _xNew;
        
        shared_ptr<vector<double>> _xOld;
        
        shared_ptr<vector<double>> _difference;
        
        bool _isInitialized;
        
        bool _throwExceptionOnMaxFailure;
        
        shared_ptr<list<double>> _residualNorms;

        
    };
    



} // LinearAlgebra

#endif //UNTITLED_ITERATIVESOLVER_H
