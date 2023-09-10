//
// Created by hal9000 on 4/12/23.
//

#ifndef UNTITLED_METRICS_H
#define UNTITLED_METRICS_H


#include "../../Node/Node.h"
#include "../../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
using namespace LinearAlgebra;

namespace Discretization{
    class Metrics {
        
    public:
        Metrics(Node* node, unsigned dimensions);
        
        // Contains the covariant base vectors at all Directions of the domain.
        // In mathematics, a set of covariant base vectors is a set of basis vectors used to define a coordinate system 
        // in a curved space. These basis vectors are tangent to the coordinate lines and transform covariantly under a 
        // change of coordinates and are expressed by the partial derivative of the natural position vector with respect
        // to the parametric coordinate system. They form the components of the covariant metric tensor, which is used
        // to measure distances and angles in the curved space.
        shared_ptr<map<Direction, NumericalVector<double>>> covariantBaseVectors;

        shared_ptr<map<Direction, NumericalVector<double>>> contravariantBaseVectors;
        
        shared_ptr<NumericalMatrix<double>>covariantTensor;

        void calculateCovariantTensor() const;

        shared_ptr<NumericalMatrix<double>>contravariantTensor;
        
        void calculateContravariantTensor() const;
        
        shared_ptr<double> jacobian;
        
        //TODO: Implement this
        void calculateJacobian();
        
        Node* node;

    };
}


#endif //UNTITLED_METRICS_H
