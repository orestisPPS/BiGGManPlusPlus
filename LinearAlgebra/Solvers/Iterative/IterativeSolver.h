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

    enum ParallelizationMethod{
        //Multi-thread solution
        vTechKickInYoo,
        //INSSSSSSSSSSSSANE GPU GAINS
        turboVTechKickInYoo,
        //:( Single thread 
        Wank
    };
    
    class IterativeSolver : public Solver {
    public:
        explicit IterativeSolver(VectorNormType normType, double tolerance = 1E-5, unsigned maxIterations = 1E4, 
                                 bool throwExceptionOnMaxFailure = true, ParallelizationMethod parallelizationMethod = Wank);
        
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
        
        VectorNormType _normType;
        
        double _tolerance;
        
        unsigned _maxIterations;
        
        shared_ptr<vector<double>> _xNew;
        
        shared_ptr<vector<double>> _xOld;
        
        shared_ptr<vector<double>> _difference;
        
        bool _isInitialized;
        
        bool _throwExceptionOnMaxFailure;
        
        shared_ptr<list<double>> _residualNorms;

        string _solverName;

        ParallelizationMethod _parallelization;
        
        virtual void _iterativeSolution();
        
        virtual void _singleThreadSolution();

        virtual void _multiThreadSolution(const unsigned short &availableThreads, const unsigned short &numberOfRows);
        
        virtual void _cudaSolution();
        
        void _printSingleThreadInitializationText();
        
        void _printMultiThreadInitializationText(unsigned short numberOfThreads);
        
        void _printCUDAInitializationText();
        
        static void _printIterationAndNorm(unsigned iteration, double norm);
        
        double _calculateNorm();
        
        void printAnalysisOutcome(unsigned totalIterations, double exitNorm, std::chrono::high_resolution_clock::time_point startTime,
                                  std::chrono::high_resolution_clock::time_point finishTime);
        
        
    };
    



} // LinearAlgebra

#endif //UNTITLED_ITERATIVESOLVER_H
