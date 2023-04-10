//
// Created by hal9000 on 1/7/23.
//

#include "Transformations.h"

namespace LinearAlgebra {
    Transformations::Transformations() = default;
    
    void Transformations::translate(vector<double> &vector, double distance1) {
        switch (vector.size()) {
            case 1:
                vector[0] += distance1;
                break;
            default:
                throw invalid_argument("Input vector should have size 1.");            
        }
    }
    
    void Transformations::translate(vector<double> &vector, double distance1, double distance2) {
        switch (vector.size()) {
            case 2:
                vector[0] += distance1;
                vector[1] += distance2;
                break;
            default:
                throw invalid_argument("Input vector should have size 2.");            
        }
    }
    
    void Transformations::translate(vector<double> &vector, double distance1, double distance2, double distance3) {
        switch (vector.size()) {
            case 3:
                vector[0] += distance1;
                vector[1] += distance2;
                vector[2] += distance3;
                break;
            default:
                throw invalid_argument("Input vector should have size 3.");            
        }
    }
    
    void Transformations::scale(vector<double> &vector, double amount1) {
        switch (vector.size()) {
            case 1:
                vector[0] *= amount1;
                break;
            default:
                throw invalid_argument("Input vector should have size 1.");            
        }
    }
    
    void Transformations::scale(vector<double> &vector, double amount1, double amount2) {
        switch (vector.size()) {
            case 2:
                vector[0] *= amount1;
                vector[1] *= amount2;
                break;
            default:
                throw invalid_argument("Input vector should have size 2.");            
        }
    }
    
    void Transformations::scale(vector<double> &vector, double amount1, double amount2, double amount3) {
        switch (vector.size()) {
            case 3:
                vector[0] *= amount1;
                vector[1] *= amount2;
                vector[2] *= amount3;
                break;
            default:
                throw invalid_argument("Input vector should have size 3.");            
        }
    }
    
    void Transformations::rotate(vector<double> &vector, double angle1) {
        if (vector.size() == 2 || vector.size() == 3)
            _rotateAroundAxis3(vector, angle1);
        else
            throw invalid_argument("Input vector should have size 2 or 3.");
    }

    void Transformations::rotate(vector<double> &vector, double angle1, double angle2, double angle3) {
        switch (vector.size()) {
            case 3:
                _rotateAroundAxis1(vector, angle1);
                _rotateAroundAxis2(vector, angle2);
                _rotateAroundAxis3(vector, angle3);
                break;
            default:
                throw invalid_argument("Input vector should have size 3.");
        }
    }
    
    void Transformations::shear(vector<double> &vector, double shear1, double shear2) {
        switch (vector.size()) {
            case 2:
                _shearPlane12(vector, shear1, shear2);
                break;
            default:
                throw invalid_argument("Input vector should have size 2.");
        }
    }
    
    void Transformations::shear(vector<double> &vector, double shear12, double shear21,
                                                        double shear13, double shear31,
                                                        double shear23, double shear32) {
        switch (vector.size()) {
            case 3:
                _shearPlane12(vector, shear12, shear21);
                _shearPlane13(vector, shear13, shear31);
                _shearPlane23(vector, shear23, shear32);
                break;
            default:
                throw invalid_argument("Input vector should have size 3.");
        }
    }
    
    void Transformations::reflect(vector<double> &vector, bool &axis1, bool &axis2, bool &axis3){
        throw runtime_error("Not implemented");
        
    }

    void Transformations::reflectAboutAxis(vector<double> &vector, Direction &axis){
        throw runtime_error("Not implemented");
        
    }

    void Transformations::reflectAboutPlane(vector<double> &vector, Direction &direction1,
                                                       Direction &direction2) {
        throw runtime_error("Not implemented");
        
    }
    
    void Transformations::_rotateAroundAxis1(vector<double> &vector, double angle) {
        std::vector<double> result(vector.size());
        angle = Utility::Calculators::degreesToRadians(angle);
        result[0] =  vector[0];
        result[1] =  vector[1] * cos(angle) + vector[2] * sin(angle);
        result[2] = -vector[1] * sin(angle) + vector[2] * cos(angle);
        vector = result;        
    }
    
    void Transformations::_rotateAroundAxis2(vector<double> &vector, double angle) {
        std::vector<double> result(vector.size());
        angle = Utility::Calculators::degreesToRadians(angle);
        result[0] = vector[0] * cos(angle) - vector[2] * sin(angle);
        result[1] = vector[1];
        result[2] = vector[0] * sin(angle) + vector[2] * cos(angle);
        vector = result;

    }
    
    void Transformations::_rotateAroundAxis3(vector<double> &vector, double angle) {
        std::vector<double> result(vector.size());
        angle = Utility::Calculators::degreesToRadians(angle);
        result[0] = vector[0] * cos(angle) - vector[1] * sin(angle);
        result[1] = vector[0] * sin(angle) + vector[1] * cos(angle);
        result[2] = vector[2];
        vector = result;

    }
    
    void Transformations::_shearPlane12(vector<double> &vector, double shear1, double shear2) {
        std::vector<double> result(vector.size());
        shear1 = Utility::Calculators::degreesToRadians(shear1);
        shear2 = Utility::Calculators::degreesToRadians(shear2);
        result[0] = vector[0] + shear1 * vector[1];
        result[1] = vector[1] + shear2 * vector[0];
        result[2] = vector[2];
        vector = result;

    }
    
    void Transformations::_shearPlane13(vector<double> &vector, double shear1, double shear3) {
        std::vector<double> result(vector.size());
        shear1 = Utility::Calculators::degreesToRadians(shear1);
        shear3 = Utility::Calculators::degreesToRadians(shear3);
        result[0] = vector[0] + shear1 * vector[2];
        result[1] = vector[1];
        result[2] = vector[2] + shear3 * vector[0];
        vector = result;
    }
    
    void Transformations::_shearPlane23(vector<double> &vector, double shear2, double shear3) {
        std::vector<double> result(vector.size());
        shear2 = Utility::Calculators::degreesToRadians(shear2);
        shear3 = Utility::Calculators::degreesToRadians(shear3);
        result[0] = vector[0];
        result[1] = vector[1] + shear2 * vector[2];
        result[2] = vector[2] + shear3 * vector[1];
        vector = result;
    }
    
    void Transformations::_reflectAboutAxis1(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] =  vector[1];
        result[2] =  vector[2];
        vector = result;
    }
    
    void Transformations::_reflectAboutAxis2(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] =  vector[0];
        result[1] = -vector[1];
        result[2] =  vector[2];
        vector = result;
    }
    
    void Transformations::_reflectAboutAxis3(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] =  vector[0];
        result[1] =  vector[1]; 
        result[2] = -vector[2];
        vector = result;
    }
    
    void Transformations::_reflectAboutPlane12(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] =  vector[0];
        result[1] = -vector[1];
        result[2] = -vector[2];
        vector = result;
    }
    
    void Transformations::_reflectAboutPlane13(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] =  vector[1];
        result[2] = -vector[2];
        vector = result;
    }
    
    void Transformations::_reflectAboutPlane23(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] = -vector[1];
        result[2] =  vector[2];
        vector = result;
    }
    
    void Transformations::_reflectAboutOrigin(vector<double> &vector) {
        std::vector<double> result(vector.size());
        result[0] = -vector[0];
        result[1] = -vector[1];
        result[2] = -vector[2];
        vector = result;
    }
} // LinearAlgebra