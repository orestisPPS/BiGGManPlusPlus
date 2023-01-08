//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_TRANSFORMATIONS_H
#define UNTITLED_TRANSFORMATIONS_H

#include <vector>
#include <cmath>
#include <limits>
#include "Array.h"
#include "../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
#include "../Utility/Calculators.h"

namespace LinearAlgebra {

    class Transformations {
        //Source1: https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg
        //Source2 :https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        public:
            Transformations();
            //Translates a vector by an input amount in the given direction.
            static vector<double> translate(vector<double> &vector, Direction direction, double distance);
            //Translates a vector by an input amount in the given 2 directions.
            static vector<double> translate(vector<double> &vector, Direction direction1, double distance1, Direction direction2, double distance2);
            //Translates a vector by an input amount in the given 3 directions.
            static vector<double> translate(vector<double> &vector, Direction direction1, double distance1, Direction direction2, double distance2, Direction direction3, double distance3);
            
            //Scales a vector by an input amount in the given direction.
            static vector<double> scale(vector<double> &vector, Direction direction, double factor);
            //Scales a vector by an input amount in the given 2 directions.
            static vector<double> scale(vector<double> &vector, Direction direction1, double factor1, Direction direction2, double factor2);
            //Scales a vector by an input amount in the given 3 directions.
            static vector<double> scale(vector<double> &vector, Direction direction1, double factor1, Direction direction2, double factor2, Direction direction3, double factor3);
            
            //Rotates a vector around the given axis by the given angle.
            static vector<double> rotate(vector<double> &vector, Direction direction, double angle);
            //Rotates a vector around the given 2 axes by the given angles.
            static vector<double> rotate(vector<double> &vector, Direction direction1, double angle1, Direction direction2, double angle2);
            //Rotates a vector around the 3 axes by the given angles.
            static vector<double> rotate(vector<double> &vector, Direction direction1, double angle1, Direction direction2, double angle2, Direction direction3, double angle3);
            
            //Shears a vector in the two available directions of the given plane
            static vector<double> shear(vector<double> &vector, Direction direction, PhysicalSpaceEntities plane, double shear1, double shear2);
            
            //Reflects a vector around the given axis.
            static vector<double> reflect(vector<double> &vector, Direction axis);
            //Reflects a vector around two given axes.
            static vector<double> reflect(vector<double> &vector, Direction axis1, Direction axis2);
            //Reflects a vector around three given axes.
            static vector<double> reflect(vector<double> &vector, Direction axis1, Direction axis2, Direction axis3);
            //Reflects a vector around the given plane.
            static vector<double> reflect(vector<double> &vector, PhysicalSpaceEntities plane);
            
            
            

        private:
            //private property that translates a vector with 3 components at all 3 directions.
            vector<double> _translate(vector<double> &vector, double translation1, double translation2, double translation3);
            //private property that scales a vector[3] at all 3 directions.
            vector<double> _scale(vector<double> &vector, double scale1, double scale2, double scale3);
            //Rotates the given array by the given angle[o] around axis 1.
            static vector<double> _rotateAroundAxis1(vector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] around axis 2.
            static vector<double> _rotateAroundAxis2(vector<double> &vector, double angle);
            //Rotates the given array by the given angle[o] around axis 3.
            static vector<double> _rotateAroundAxis3(vector<double> &vector, double angle);
            //Shears the given array by the given angles[o] in direction 1 and 2 in the 1-2 plane.
            static vector<double> _shearPlane12(vector<double> &vector, double angle1, double angle2);
            //Shears the given array by the given angles[o] in direction 1 and 3 in the 1-3 plane.
            static vector<double> _shearPlane13(vector<double> &vector, double angle1, double angle3);
            //Shears the given array by the given angles[o] in direction 2 and 3 in the 2-3 plane.
            static vector<double> _shearPlane23(vector<double> &vector, double angle2, double angle3);
            //Reflects the given array about axis1.
            static vector<double> _reflectAboutAxis1(vector<double> &vector);
            //Reflects the given array about axis2.
            static vector<double> _reflectAboutAxis2(vector<double> &vector);
            //Reflects the given array about axis3.
            static vector<double> _reflectAboutAxis3(vector<double> &vector);
            //Reflects the given array about the 1-2 plane.
            static vector<double> _reflectAboutPlane12(vector<double> &vector);
            //Reflects the given array about the 1-3 plane.
            static vector<double> _reflectAboutPlane13(vector<double> &vector);
            //Reflects the given array about the 2-3 plane.
            static vector<double> _reflectAboutPlane23(vector<double> &vector);
            //Reflects the given array about the origin.
            static vector<double> _reflectAboutOrigin(vector<double> &vector);
            
            
            
            
            

    };

} // LinearAlgebra

#endif //UNTITLED_TRANSFORMATIONS_H
