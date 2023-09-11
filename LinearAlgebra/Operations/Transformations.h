//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_TRANSFORMATIONS_H
#define UNTITLED_TRANSFORMATIONS_H

#include <vector>
#include <cmath>
#include <limits>
#include "../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../PositioningInSpace/PhysicalSpaceEntities/PhysicalSpaceEntity.h"

namespace LinearAlgebra {

    class Transformations {
        //Source1: https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg
        //Source2 :https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialighTransformationMatrices.html
        public:
            Transformations();
            
            //Translates the input vector in the 1d space.
            //distance1 : Amount to translate vector in axis 1.
            static void translate(NumericalVector<double> &vector, double distance1);
            
            //Translates the input vector in the 2d space.
            //distance1,2 : Amount to translate vector in the 1-2 plane.
            static void translate(NumericalVector<double> &vector, double distance1, double distance2);
            
            //Translates the input vector in the 3d space.
            //distance1,2,3 : Amount to translate vector in the 1-2-3 volume.
            static void translate(NumericalVector<double> &vector, double distance1, double distance2, double distance3);
            
            //Scales the input vector in the 1d space.
            //amount1 : Amount to scale vector in axis 1.
            static void scale(NumericalVector<double> &vector, double amount1);
            
            //Scales the input vector in the 2d space.
            //amount1,2 : Amount to scale vector in the 1-2 plane.
            static void scale(NumericalVector<double> &vector, double amount1, double amount2);
            
            //Scales the input vector in the 3d space.
            //amount1,2,3 : Amount to scale vector in the 1-2-3 volume.
            static void scale(NumericalVector<double> &vector, double amount1, double amount2, double amount3);
            
            //Rotates the 2D input vector around the axis normal to the plane it lies.
            //angle : Angle to rotate.
            static void rotate(NumericalVector<double> &vector, double angle);
                        
            //Rotates the input vector in the 3d space by an input angle [deg] around all three axis.
            //angle1,2,3 : Amount to rotate around axis 1,2,3 respectively.
            static void rotate(NumericalVector<double> &vector, double angle1, double angle2, double angle3);
            
            //Shears the input vector in the 2d space in the plane it lies.
            //angle12 : First angle to shear in the 1 - 2 plane.
            //angle21 : Second angle to shear in the 1 - 2 plane.
            static void shear(NumericalVector<double> &vector, double angle12, double angle21);
            
            //Shears the input vector in the 3d space by an input angle in the given plane.
            //angle12,13,23 : Amount to shear in the 12,23,13 planes respectively.
            static void shear(NumericalVector<double> &vector, double angle12, double angle21,
                                                      double angle13, double angle31,
                                                      double angle23, double angle32);
                        
            //Reflects the input vector in the 3d space.
            //Axis1,2,3 : true if the vector should be reflected in the direction of axis 1,2,3 respectively. false otherwise.
            static void reflect(NumericalVector<double> &vector, bool &axis1, bool &axis2, bool &axis3);
            
            //Reflects the input vector around the given plane.
            static void reflectAboutAxis(NumericalVector<double> &vector, Direction &axis);
            
            //Reflects the input vector around the given plane.
            static void reflectAboutPlane(NumericalVector<double> &vector, Direction &direction1, Direction &direction2);
            
            
        private:
            //Rotates the given array by the given angle[o] around axis 1.
            static void _rotateAroundAxis1(NumericalVector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] around axis 2.
            static void _rotateAroundAxis2(NumericalVector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] around axis 3.
            static void _rotateAroundAxis3(NumericalVector<double> &vector, double angle);
            //Shears the given array by the given angles[o] in direction 1 and 2 in the 1-2 plane.
            static void _shearPlane12(NumericalVector<double> &vector, double angle1, double angle2);
            //Shears the given array by the given angles[o] in direction 1 and 3 in the 1-3 plane.
            static void _shearPlane13(NumericalVector<double> &vector, double angle1, double angle3);
            //Shears the given array by the given angles[o] in direction 2 and 3 in the 2-3 plane.
            static void _shearPlane23(NumericalVector<double> &vector, double angle2, double angle3);
            //Reflects the given array about axis1.
            static void _reflectAboutAxis1(NumericalVector<double> &vector);
            //Reflects the given array about axis2.
            static void _reflectAboutAxis2(NumericalVector<double> &vector);
            //Reflects the given array about axis3.
            static void _reflectAboutAxis3(NumericalVector<double> &vector);
            //Reflects the given array about the 1-2 plane.
            static void _reflectAboutPlane12(NumericalVector<double> &vector);
            //Reflects the given array about the 1-3 plane.
            static void _reflectAboutPlane13(NumericalVector<double> &vector);
            //Reflects the given array about the 2-3 plane.
            static void _reflectAboutPlane23(NumericalVector<double> &vector);
            //Reflects the given array about the origin.
            static void _reflectAboutOrigin(NumericalVector<double> &vector);
            
    };

} // LinearAlgebra

#endif //UNTITLED_TRANSFORMATIONS_H
