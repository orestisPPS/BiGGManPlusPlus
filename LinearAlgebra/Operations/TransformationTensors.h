//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_TRANSFORMATIONTENSORS_H
#define UNTITLED_TRANSFORMATIONTENSORS_H

#include <vector>
#include <cmath>
#include <limits>
#include "../Array/Array.h"
#include "../../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
#include "../../Utility/Calculators.h"

using namespace std;

namespace LinearAlgebra {

            //Source1: https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg
            //Source2 :https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
            // The transformation matrix that translates a vector in direction one by an input amount
            static Array<double> translationOneTensor(double translationOne) {
                Array<double> translationOneTensor(3, 3);
                translationOneTensor.populateElement(0, 0, 1);
                translationOneTensor.populateElement(0, 2, translationOne);
                translationOneTensor.populateElement(1, 1, 1);
                translationOneTensor.populateElement(2, 2, 1);
                return translationOneTensor;
            }

            // The transformation matrix that translates a vector in direction two by an input amount
            static Array<double> translationTwoTensor(double translationTwo) {
                Array<double> translationTwoTensor(3, 3);
                translationTwoTensor.populateElement(0, 0, 1);
                translationTwoTensor.populateElement(1, 1, 1);
                translationTwoTensor.populateElement(1, 2, translationTwo);
                translationTwoTensor.populateElement(2, 2, 1);
                return translationTwoTensor;
            }

            // The transformation matrix that translates a vector in direction three by an input amount
            static Array<double> translationThreeTensor(double translationThree) {
                Array<double> translationThreeTensor(3, 3);
                translationThreeTensor.populateElement(0, 0, 1);
                translationThreeTensor.populateElement(1, 1, 1);
                translationThreeTensor.populateElement(2, 2, 1);
                translationThreeTensor.populateElement(2, 0, translationThree);
                return translationThreeTensor;
            }

            // The transformation tensor that scales a vector in direction one by an input amount
            static Array<double> scalingOneTensor(double scalingOne) {
                Array<double> scalingTensor(3, 3);
                scalingTensor.populateElement(0, 0, scalingOne);
                scalingTensor.populateElement(1, 1, 1);
                scalingTensor.populateElement(2, 2, 1);
                return scalingTensor;
            }

            // The transformation tensor that scales a vector in direction two by an input amount
            static Array<double> scalingTwoTensor(double scalingTwo) {
                Array<double> scalingTensor(3, 3);
                scalingTensor.populateElement(0, 0, 1);
                scalingTensor.populateElement(1, 1, scalingTwo);
                scalingTensor.populateElement(2, 2, 1);
                return scalingTensor;
            }

            // The transformation tensor that scales a vector in direction three by an input amount
            static Array<double> scalingThreeTensor(double scalingThree) {
                Array<double> scalingTensor(3, 3);
                scalingTensor.populateElement(0, 0, 1);
                scalingTensor.populateElement(1, 1, 1);
                scalingTensor.populateElement(2, 2, scalingThree);
                return scalingTensor;
            }

            // The transformation tensor that rotates a vector around axis 1 in the 2-3 plane by an input angle
            static Array<double> rotationTensorOne(double angleInDegrees) {
                auto angleInRadians = Utility::Calculators::degreesToRadians(angleInDegrees);
                Array<double> rotationTensorOne(3, 3);
                rotationTensorOne.populateElement(0, 0, 1);
                rotationTensorOne.populateElement(1, 1, cos(angleInRadians));
                rotationTensorOne.populateElement(1, 2, -sin(angleInRadians));
                rotationTensorOne.populateElement(2, 1, sin(angleInRadians));
                rotationTensorOne.populateElement(2, 2, cos(angleInRadians));
                return rotationTensorOne;
            }

            // The transformation tensor that rotates a vector around axis 2 in the 1-3 plane by an input angle
            static Array<double> rotationTensorTwo(double angleInDegrees) {
                auto angleInRadians = Utility::Calculators::degreesToRadians(angleInDegrees);
                Array<double> rotationTensorTwo(3, 3);
                rotationTensorTwo.populateElement(0, 0, cos(angleInRadians));
                rotationTensorTwo.populateElement(0, 2, sin(angleInRadians));
                rotationTensorTwo.populateElement(1, 1, 1);
                rotationTensorTwo.populateElement(2, 0, -sin(angleInRadians));
                rotationTensorTwo.populateElement(2, 2, cos(angleInRadians));
                return rotationTensorTwo;
            }

            // The transformation tensor that rotates a vector around axis 3 in the 1-2 plane by an input angle
            static Array<double> rotationTensorThree(double angleInDegrees) {
                auto angleInRadians = Utility::Calculators::degreesToRadians(angleInDegrees);
                Array<double> rotationTensorThree(3, 3);
                rotationTensorThree.populateElement(0, 0, cos(angleInRadians));
                rotationTensorThree.populateElement(0, 1, -sin(angleInRadians));
                rotationTensorThree.populateElement(1, 0, sin(angleInRadians));
                rotationTensorThree.populateElement(1, 1, cos(angleInRadians));
                rotationTensorThree.populateElement(2, 2, 1);
                return rotationTensorThree;
            }

            // The transformation tensor that shears a vector in direction 1 in the 1-2 plane by an input angle
            static Array<double> shearTensorOne_planeOneTwo(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                Array<double> shearTensorOne_planeOneTwo(3, 3);
                shearTensorOne_planeOneTwo.populateElement(0, 0, 1);
                shearTensorOne_planeOneTwo.populateElement(0, 1, ::tan(shearAngleInDegrees));
                shearTensorOne_planeOneTwo.populateElement(1, 1, 1);
                shearTensorOne_planeOneTwo.populateElement(2, 2, 1);
                return shearTensorOne_planeOneTwo;
            }

            // The transformation tensor that shears a vector in direction 2 in the 1-2 plane by an input angle
            static Array<double> shearTensorTwo_planeOneTwo(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                Array<double> shearTensorTwo_planeOneTwo(3, 3);
                shearTensorTwo_planeOneTwo.populateElement(0, 0, 1);
                shearTensorTwo_planeOneTwo.populateElement(1, 0, ::tan(shearAngleInDegrees));
                shearTensorTwo_planeOneTwo.populateElement(1, 1, 1);
                shearTensorTwo_planeOneTwo.populateElement(2, 2, 1);
                return shearTensorTwo_planeOneTwo;
            }

            // The transformation tensor that shears a vector in direction 1 in the 1-3 plane by an input angle
            static Array<double> shearTensorOne_planeOneThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                Array<double> shearTensorOne_planeOneThree(3, 3);
                shearTensorOne_planeOneThree.populateElement(0, 0, 1);
                shearTensorOne_planeOneThree.populateElement(0, 2, ::tan(shearAngleInDegrees));
                shearTensorOne_planeOneThree.populateElement(1, 1, 1);
                shearTensorOne_planeOneThree.populateElement(2, 2, 1);
                return shearTensorOne_planeOneThree;
            }

            // The transformation tensor that shears a vector in direction 3 in the 1-3 plane by an input angle
            static Array<double> shearTensorThree_planeOneThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                Array<double> shearTensorThree_planeOneThree(3, 3);
                shearTensorThree_planeOneThree.populateElement(0, 0, 1);
                shearTensorThree_planeOneThree.populateElement(2, 0, ::tan(shearAngleInDegrees));
                shearTensorThree_planeOneThree.populateElement(1, 1, 1);
                shearTensorThree_planeOneThree.populateElement(2, 2, 1);
                return shearTensorThree_planeOneThree;
            }

            // The transformation tensor that shears a vector in direction 2 in the 2-3 plane by an input angle
            static Array<double> shearTensorTwo_planeTwoThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                Array<double> shearTensorTwo_planeTwoThree(3, 3);
                shearTensorTwo_planeTwoThree.populateElement(0, 0, 1);
                shearTensorTwo_planeTwoThree.populateElement(1, 1, 1);
                shearTensorTwo_planeTwoThree.populateElement(1, 2, ::tan(shearAngleInDegrees));
                shearTensorTwo_planeTwoThree.populateElement(2, 2, 1);
                return shearTensorTwo_planeTwoThree;
            }

            // The transformation tensor that shears a vector in direction 3 in the 2-3 plane by an input angle
            static static Array<double> shearTensorThree_planeTwoThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                Array<double> shearTensorThree_planeTwoThree(3, 3);
                shearTensorThree_planeTwoThree.populateElement(0, 0, 1);
                shearTensorThree_planeTwoThree.populateElement(1, 1, 1);
                shearTensorThree_planeTwoThree.populateElement(2, 1, ::tan(shearAngleInDegrees));
                shearTensorThree_planeTwoThree.populateElement(2, 2, 1);
                return shearTensorThree_planeTwoThree;
            }

            // The transformation tensor that reflects about  axis 1  in the 1-2 plane
            static Array<double> ReflectionTensorOne_planeOneTwo() {
                Array<double> reflectionTensorOne_planeOneTwo(3, 3);
                reflectionTensorOne_planeOneTwo.populateElement(0, 0, 1);
                reflectionTensorOne_planeOneTwo.populateElement(1, 1, -1);
                reflectionTensorOne_planeOneTwo.populateElement(2, 2, 1);
                return reflectionTensorOne_planeOneTwo;
            }

            // The transformation tensor that reflects about  axis 2  in the 1-2 plane
            static Array<double> ReflectionTensorTwo_planeOneTwo() {
                Array<double> reflectionTensorTwo_planeOneTwo(3, 3);
                reflectionTensorTwo_planeOneTwo.populateElement(0, 0, -1);
                reflectionTensorTwo_planeOneTwo.populateElement(1, 1, 1);
                reflectionTensorTwo_planeOneTwo.populateElement(2, 2, 1);
                return reflectionTensorTwo_planeOneTwo;
            }

            // The transformation tensor that reflects about  axis 1  in the 1-3 plane
            static Array<double> ReflectionTensorOne_planeOneThree() {
                Array<double> reflectionTensorOne_planeOneThree(3, 3);
                reflectionTensorOne_planeOneThree.populateElement(0, 0, 1);
                reflectionTensorOne_planeOneThree.populateElement(1, 1, 1);
                reflectionTensorOne_planeOneThree.populateElement(2, 2, -1);
                return reflectionTensorOne_planeOneThree;
            }

            // The transformation tensor that reflects about  axis 3  in the 1-3 plane
            static Array<double> ReflectionTensorThree_planeOneThree() {
                Array<double> reflectionTensorThree_planeOneThree(3, 3);
                reflectionTensorThree_planeOneThree.populateElement(0, 0, -1);
                reflectionTensorThree_planeOneThree.populateElement(1, 1, 1);
                reflectionTensorThree_planeOneThree.populateElement(2, 2, 1);
                return reflectionTensorThree_planeOneThree;
            }

            // The transformation tensor that reflects about  axis 2  in the 2-3 plane
            static Array<double> ReflectionTensorTwo_planeTwoThree() {
                Array<double> reflectionTensorTwo_planeTwoThree(3, 3);
                reflectionTensorTwo_planeTwoThree.populateElement(0, 0, 1);
                reflectionTensorTwo_planeTwoThree.populateElement(1, 1, 1);
                reflectionTensorTwo_planeTwoThree.populateElement(2, 2, -1);
                return reflectionTensorTwo_planeTwoThree;
            }

            // The transformation tensor that reflects about  axis 3  in the 2-3 plane
            static Array<double> ReflectionTensorThree_planeTwoThree() {
                Array<double> reflectionTensorThree_planeTwoThree(3, 3);
                reflectionTensorThree_planeTwoThree.populateElement(0, 0, 1);
                reflectionTensorThree_planeTwoThree.populateElement(1, 1, -1);
                reflectionTensorThree_planeTwoThree.populateElement(2, 2, 1);
                return reflectionTensorThree_planeTwoThree;
            }
        }






/*        switch (array.size()) {
            case 2:
                rotatedArray.push_back(array[0] * cos(angleInRadians) - array[1] * sin(angleInRadians));
                rotatedArray.push_back(array[0] * sin(angleInRadians) + array[1] * cos(angleInRadians));
                break;
            case 3:
                rotatedArray.push_back(array[0] * cos(angleInRadians) - array[1] * sin(angleInRadians));
                rotatedArray.push_back(array[0] * sin(angleInRadians) + array[1] * cos(angleInRadians));
                rotatedArray.push_back(array[2]);
                break;
            default:
                //return NaN vector
                for (int i = 0; i < array.size(); i++) {
                    rotatedArray.push_back(numeric_limits<double>::quiet_NaN());
                }
        }*/





#endif //UNTITLED_TRANSFORMATIONTENSORS_H

/*static vector<double> Shear(vector<double> &array, double shearAngleInDegrees, PositioningInSpace::Direction axis){
    vector<double> shearedArray;
    shearAngleInDegrees = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
    
    switch (array.size()) {
        case 2:
            if (axis == 0)
                shearedArray.push_back()
        case 3:
            if (axis == 0) {
                shearedArray.push_back(array[0]);
                shearedArray.push_back(array[1] + array[0] * tan(shearAngleInDegrees));
                shearedArray.push_back(array[2]);
            } else if (axis == 1) {
                shearedArray.push_back(array[0] + array[1] * tan(shearAngleInDegrees));
                shearedArray.push_back(array[1]);
                shearedArray.push_back(array[2]);
            } else if (axis == 2) {
                shearedArray.push_back(array[0]);
                shearedArray.push_back(array[1]);
                shearedArray.push_back(array[2] + array[0] * tan(shearAngleInDegrees));
            }
            break;
        default:
            //return NaN vector
            for (int i = 0; i < array.size(); i++) {
                shearedArray.push_back(numeric_limits<double>::quiet_NaN());
            }
    }
    return shearedArray;
}*/