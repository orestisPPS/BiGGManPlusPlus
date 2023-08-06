//
// Created by hal9000 on 8/4/23.
//

#ifndef UNTITLED_POWERMETHOD_H
#define UNTITLED_POWERMETHOD_H

#include <random>
#include "../Array/Array.h"
#include "../Norms/VectorNorm.h"
#include "../Operations/MultiThreadVectorOperations.h"
#include "../../Utility/Exporters/Exporters.h"
#include "../Operations/VectorOperations.h"
namespace LinearAlgebra {



    class PowerMethod {
    public:

        enum ParallelizationMethod{
            //Multi-thread solution
            vTechKickInYoo,
            //INSSSSSSSSSSSSANE GPU GAINS
            turboVTechKickInYoo,
            //:( Single thread 
            Wank
        };

        PowerMethod(unsigned short numberOfEigenvalues, unsigned maxIterations, VectorNormType normType = L2,
                                  double tolerance = 1E-5, ParallelizationMethod parallelizationMethod = Wank );

        void calculateDominantEigenValue();
        
        void setMatrix(const shared_ptr<Array<double>>& matrix);

    private:

        void _singleThreadSolution();

        void _singleThreadOrthogonalization(shared_ptr<vector<double>> &vectorToOrthogonalize);\

        void _singleThreadCompleteOrthogonalization(shared_ptr<vector<double>> vectorToOrthogonalize);

        void _multiThreadSolution();

        void _cudaSolution();

        unsigned int _numberOfEigenvalues;

        shared_ptr<Array<double>> _matrix;

        shared_ptr<map<unsigned, shared_ptr<vector<double>>>> _lanczosVectors;

        shared_ptr<vector<double>> _vectorNew;

        shared_ptr<vector<double>> _vectorOld;
        
        shared_ptr<vector<double>> _difference;

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

#endif //UNTITLED_POWERMETHOD_H
