//
// Created by hal9000 on 4/18/23.
//

#ifndef UNTITLED_VECTORNORM_H
#define UNTITLED_VECTORNORM_H

#include <map>
#include <cmath>
#include <functional>
#include "../Array.h"

namespace LinearAlgebra {

    enum VectorNormType {
        // L1 (Manhattan / Taxicab) norm
        // The sum of the absolute values of the vectors' components.
        // For a vector x with n components, the L1 norm is denoted as ||x||1 and defined as:
        // ||x||1 = |x₁| + |x₂| + ... + |xₙ|
        L1,
        
        // L2 (Euclidean) norm
        // The square root of the sum of the squares of the vectors' components.
        // For a vector x with n components, the L2 norm is denoted as ||x||2 and defined as:
        // ||x||2 = √(x₁² + x₂² + ... + xₙ²)
        L2,
        
        // L∞ (Chebyshev) norm
        // The maximum absolute value of the vectors' components.
        // For a vector x with n components, the L∞ norm is denoted as ||x||∞ and defined as:
        // ||x||∞ = max(|x₁|, |x₂|, ..., |xₙ|)
        LInf,
        
        // Lp norm
        // The pth root of the sum of the pth powers of the vectors' components.    
        // For a vector x with n components, the Lp norm is denoted as ||x||p and defined as:
        // ||x||p = (|x₁|^p + |x₂|^p + ... + |xₙ|^p)^(1/p)
        Lp,
        
/*        //Frobenius (Euclidean for matrices) norm
        // Defined only for Array class. 
        // The square root of the sum of the squares of the matrices' components.
        // For a matrix A with m rows and n columns, the Frobenius norm is denoted as ||A||F and defined as:
        // ||A||F = √(A₁₁² + A₁₂² + ... + Aₘₙ²)
        Frobenius*/
    };
    
    class VectorNorm {
        
    public:
        
        VectorNorm(vector<double>* vector, VectorNormType normType);
        
        VectorNorm(vector<double>* vector, VectorNormType normType, unsigned order);
        
        VectorNormType & type();
        
        double value() const;
        static double _calculateLInfNorm(vector<double>* vector);
    private: 
        
        VectorNormType _normType;

/*        //double (*func)(vector<double>* vector);
        map<VectorNormType, double (vector<double>* vector)> _normTypeToFunction = {
                {L1, _calculateL1Norm},
                {L2, _calculateL2Norm},
                {LInf, _calculateLInfNorm},
                {Lp, _calculateLpNorm}
        };*/
        
        
        
        double _value;
        
        double _calculateL1Norm(vector<double>* vector);

        double _calculateL2Norm(vector<double>* vector);

        //static double _calculateLInfNorm(vector<double>* vector);

        static double _calculateLpNorm(vector<double>* vector, double order);
    };
} // LinearAlgebra

#endif //UNTITLED_VECTORNORM_H
