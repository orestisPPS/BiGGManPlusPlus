//
// Created by hal9000 on 8/3/23.
//

#ifndef UNTITLED_LANCZOSEIGENDECOMPOSITION_H
#define UNTITLED_LANCZOSEIGENDECOMPOSITION_H
#include <random>
#include "../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../Utility/Exporters/Exporters.h"
namespace LinearAlgebra {


    
    class LanczosEigenDecomposition {
    public:

        enum ParallelizationMethod{
            //Multi-thread solution
            vTechKickInYoo,
            //INSSSSSSSSSSSSANE GPU GAINS
            turboVTechKickInYoo,
            //:( Single thread 
            Wank
        };

        LanczosEigenDecomposition(unsigned short numberOfEigenvalues, unsigned lanczosIterations, VectorNormType normType = L2,
                                  double tolerance = 1E-5, ParallelizationMethod parallelizationMethod = Wank );
        
        void calculateEigenvalues();
        
        void setMatrix(const shared_ptr<NumericalMatrix<double>>& matrix);
        
    private:
        
        void _singleThreadSolution();
        
        void _singleThreadOrthogonalization(shared_ptr<NumericalVector<double>> &vectorToOrthogonalize);\
        
        void _singleThreadCompleteOrthogonalization(shared_ptr<NumericalVector<double>> vectorToOrthogonalize);
        
        void _multiThreadSolution();
        
        void _cudaSolution();
        
        unsigned int _numberOfEigenvalues;
        
        shared_ptr<NumericalMatrix<double>> _matrix;
        
        shared_ptr<map<unsigned, shared_ptr<NumericalVector<double>>>> _lanczosVectors;
        
        shared_ptr<NumericalVector<double>> _lanczosVectorNew;

        shared_ptr<NumericalVector<double>> _lanczosVectorOld;
        
        shared_ptr<NumericalVector<double>> _T_diagonal;
        
        shared_ptr<NumericalVector<double>> _T_subDiagonal;
        
        shared_ptr<NumericalMatrix<double>> _T_matrix;
        
        double _alpha, _beta;

        VectorNormType _normType;

        double _tolerance;

        unsigned _iteration;

        unsigned _maxIterations;

        double _exitNorm;
        
        ParallelizationMethod _parallelizationMethod;
        
        bool _vectorsInitialized;
        
        bool _matrixSet;

        void _initializeVectors();
        
        static void _printSingleThreadInitializationText();

        static void _printMultiThreadInitializationText(unsigned short numberOfThreads);

        void _printCUDAInitializationText();

        void _printIterationAndNorm(unsigned displayFrequency = 100) const;

        double _calculateNorm();

        void printAnalysisOutcome(unsigned totalIterations, double exitNorm, std::chrono::high_resolution_clock::time_point startTime,
                                  std::chrono::high_resolution_clock::time_point finishTime) const;
    };

} // LinearAlgebra

#endif //UNTITLED_LANCZOSEIGENDECOMPOSITION_H
