//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_ITERATIVESOLVER_H
#define UNTITLED_ITERATIVESOLVER_H

#include <thread>
#include "../Solver.h"
#include "../../AnalysisLinearSystemInitializer.h"
#include "../../Norms/VectorNorm.h"
#include "../../Operations/MultiThreadVectorOperations.h"   
#include "../../ParallelizationMethods.h"
using LinearAlgebra::ParallelizationMethod;

namespace LinearAlgebra {
    
    class IterativeSolver : public Solver {
    public:
        explicit IterativeSolver(VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4, 
                                 bool throwExceptionOnMaxFailure = true, ParallelizationMethod parallelizationMethod = SingleThread);
        
        ~IterativeSolver();

        void setTolerance(double tolerance);
        
        const double& getTolerance() const;
        
        void setMaxIterations(unsigned maxIterations);
        
        const unsigned& getMaxIterations() const;
        
        void setNormType(VectorNormType normType);
        
        const VectorNormType& getNormType() const;
        
        void solve() override;
        
    protected:
        
        VectorNormType _normType;
        
        double _tolerance;
        
        unsigned _iteration;
        
        unsigned _maxIterations;
        
        double _exitNorm;
        
        shared_ptr<vector<double>> _xNew;
        
        shared_ptr<vector<double>> _xOld;

        shared_ptr<vector<double>> _difference;
        
        bool _throwExceptionOnMaxFailure;
        
        shared_ptr<list<double>> _residualNorms;

        string _solverName;

        ParallelizationMethod _parallelization;

        
        void setInitialSolution(shared_ptr<vector<double>> initialSolution) override;

        void setInitialSolution(double initialValue) override;
        
        virtual void _iterativeSolution();
        
        virtual void _singleThreadSolution();

        virtual void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows);
        
        virtual void _cudaSolution();
        
        void _printSingleThreadInitializationText();
        
        void _printMultiThreadInitializationText(unsigned short numberOfThreads);
        
        void _printCUDAInitializationText();
        
        void _printIterationAndNorm(unsigned displayFrequency = 100) const;
        
        double _calculateNorm();
        
        void printAnalysisOutcome(unsigned totalIterations, double exitNorm, std::chrono::high_resolution_clock::time_point startTime,
                                  std::chrono::high_resolution_clock::time_point finishTime) const;
        
        
    };
    



} // LinearAlgebra

#endif //UNTITLED_ITERATIVESOLVER_H
