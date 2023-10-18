//
// Created by hal9000 on 5/5/23.
//

#ifndef UNTITLED_SPATIALPDEPROPERTIES_H
#define UNTITLED_SPATIALPDEPROPERTIES_H
#include "../../LinearAlgebra/ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "PDEProperties/SecondOrderLinearPDEProperties.h"

namespace MathematicalEntities {

    /**
     * @struct SpatialScalarFieldPDEProperties
     * @brief Contains coefficients for spatial derivatives of a scalar field in a PDE.
     * 
     * This struct stores coefficients related to the spatial derivatives of a scalar field PDE.
     * Second order derivatives are stored in a NumericalMatrix object ptr, first order derivatives are stored in a NumericalVector object ptr,
     * and the zeroth order as well as the source terms are stored in a double ptr.
     */
    struct SpatialScalarFieldPDEProperties {
        /// Coefficients for the first order spatial derivatives.
        unique_ptr<NumericalVector<double>> firstOrderCoefficients;
        /// Coefficients matrix for the second order spatial derivatives.
        unique_ptr<NumericalMatrix<double>> secondOrderCoefficients;
        /// Coefficient for the zeroth order term (i.e., the field value itself).
        unique_ptr<double> zerothOrderCoefficient;
        /// Source term or the inhomogeneous term in the PDE.
        unique_ptr<double> sourceTerm;
    };

    /**
     * @struct SpatialVectorFieldPDEProperties
     * @brief Contains coefficients for spatial derivatives of a vector field in a PDE.
     * 
     * Designed to hold coefficients related to the spatial derivatives of a vector field PDE.
     * Second and first order derivatives are stored in NumericalMatrix object ptr, zeroth order derivatives
     * are stored in a NumericalVector object ptr, and the source term is stored in a double ptr.
     */
    struct SpatialVectorFieldPDEProperties {
        /// Coefficients matrix for the first order spatial derivatives.
        unique_ptr<NumericalMatrix<double>> firstOrderCoefficients;
        /// Coefficients matrix for the second order spatial derivatives.
        unique_ptr<NumericalMatrix<double>> secondOrderCoefficients;
        /// Coefficients for the zeroth order term (i.e., the field value itself).
        unique_ptr<NumericalVector<double>> zerothOrderCoefficient;
        /// Source term or the inhomogeneous term in the PDE.
        unique_ptr<double> sourceTerm;
    };

    /**
     * @struct TemporalVectorFieldPDECoefficients
     * @brief Holds coefficients for temporal derivatives of a vector field in a PDE.
     * 
     * Captures coefficients tied to the temporal derivatives in a vector field PDE. 
     * Contains NumericalVector object ptrs for the first and second order temporal derivatives.
     */
    struct TemporalVectorFieldPDECoefficients {
        /// Coefficients for the first order temporal derivatives.
        unique_ptr<NumericalVector<double>> firstOrderCoefficients;
        /// Coefficients for the second order temporal derivatives.
        unique_ptr<NumericalVector<double>> secondOrderCoefficients;
    };

    /**
     * @struct TemporalScalarFieldPDECoefficients
     * @brief Holds coefficients for temporal derivatives of a scalar field in a PDE.
     * 
     * Designed to keep coefficients related to the temporal part of a scalar field PDE. 
     * Contains double ptrs for the first and second order temporal derivatives.
     */
    struct TemporalScalarFieldPDECoefficients {
        /// Coefficient for the first order temporal derivative.
        unique_ptr<double> firstOrderCoefficient;
        /// Coefficient for the second order temporal derivative.
        unique_ptr<double> secondOrderCoefficient;
    };

    
} // MathematicalEntities

#endif //UNTITLED_SPATIALPDEPROPERTIES_H







