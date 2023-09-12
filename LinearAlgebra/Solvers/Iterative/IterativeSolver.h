//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_ITERATIVESOLVER_H
#define UNTITLED_ITERATIVESOLVER_H

#include <thread>
#include "../Solver.h"
#include "../../AnalysisLinearSystemInitializer.h"

using LinearAlgebra::ParallelizationMethod;

namespace LinearAlgebra {
    
    class IterativeSolver : public Solver {
    public:
        explicit IterativeSolver(double tolerance = 1E-5, unsigned maxIterations = 1E4, VectorNormType normType = L2,
                                 unsigned userDefinedThreads = 0, bool printOutput = true, bool throwExceptionOnMaxFailure = true);
        
        ~IterativeSolver();

        void setTolerance(double tolerance);
        
        const double& getTolerance() const;
        
        void setMaxIterations(unsigned maxIterations);
        
        const unsigned& getMaxIterations() const;
        
        void setNormType(VectorNormType normType);
        
        const VectorNormType& getNormType() const;
        
        void solve() override;
        
    protected:
        double _tolerance;

        unsigned _maxIterations;

        VectorNormType _normType;
        
        unsigned _userDefinedThreads;
        
        unsigned _iteration;
        
        
        double _exitNorm;
        
        shared_ptr<NumericalVector<double>> _xNew;
        
        shared_ptr<NumericalVector<double>> _xOld;

        shared_ptr<NumericalVector<double>> _difference;
        
        bool _throwExceptionOnMaxFailure;
        
        bool _printOutput;
        
        shared_ptr<list<double>> _residualNorms;

        string _solverName;

        void _iterativeSolution();

        void setInitialSolution(shared_ptr<NumericalVector<double>> initialSolution) override;

        void setInitialSolution(double initialValue) override;
        
        virtual void _performMethodIteration();
        
        virtual void _cudaSolution();
        
        void _printInitializationText();
        
        void _printCUDAInitializationText();
        
        void _printIterationAndNorm(unsigned displayFrequency = 100) const;
        
        void printAnalysisOutcome(unsigned totalIterations, double exitNorm, std::chrono::high_resolution_clock::time_point startTime,
                                  std::chrono::high_resolution_clock::time_point finishTime) const;
        
        
    };
    



} // LinearAlgebra

#endif //UNTITLED_ITERATIVESOLVER_H
