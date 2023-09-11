//
// Created by hal9000 on 1/7/23.
//

#ifndef UNTITLED_TRANSFORMATIONTENSORS_H
#define UNTITLED_TRANSFORMATIONTENSORS_H

#include <vector>
#include <cmath>
#include <limits>
#include "../ContiguousMemoryNumericalArrays/NumericalMatrix/NumericalMatrix.h"
#include "../../PositioningInSpace/DirectionsPositions.h"
using namespace PositioningInSpace;
#include "../../Utility/Calculators.h"

using namespace std;

namespace LinearAlgebra {

            //Source1: https://en.wikipedia.org/wiki/Transformation_matrix#/media/File:2D_affine_transformation_matrix.svg
            //Source2 :https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
            // The transformation matrix that translates a vector in direction one by an input amount
            static NumericalMatrix<double> translationOneTensor(double translationOne) {
                NumericalMatrix<double> translationOneTensor(3, 3);
                translationOneTensor.setElement(0, 0, 1);
                translationOneTensor.setElement(0, 2, translationOne);
                translationOneTensor.setElement(1, 1, 1);
                translationOneTensor.setElement(2, 2, 1);
                return translationOneTensor;
            }

            // The transformation matrix that translates a vector in direction two by an input amount
            static NumericalMatrix<double> translationTwoTensor(double translationTwo) {
                NumericalMatrix<double> translationTwoTensor(3, 3);
                translationTwoTensor.setElement(0, 0, 1);
                translationTwoTensor.setElement(1, 1, 1);
                translationTwoTensor.setElement(1, 2, translationTwo);
                translationTwoTensor.setElement(2, 2, 1);
                return translationTwoTensor;
            }

            // The transformation matrix that translates a vector in direction three by an input amount
            static NumericalMatrix<double> translationThreeTensor(double translationThree) {
                NumericalMatrix<double> translationThreeTensor(3, 3);
                translationThreeTensor.setElement(0, 0, 1);
                translationThreeTensor.setElement(1, 1, 1);
                translationThreeTensor.setElement(2, 2, 1);
                translationThreeTensor.setElement(2, 0, translationThree);
                return translationThreeTensor;
            }

            // The transformation tensor that scales a vector in direction one by an input amount
            static NumericalMatrix<double> scalingOneTensor(double scalingOne) {
                NumericalMatrix<double> scalingTensor(3, 3);
                scalingTensor.setElement(0, 0, scalingOne);
                scalingTensor.setElement(1, 1, 1);
                scalingTensor.setElement(2, 2, 1);
                return scalingTensor;
            }

            // The transformation tensor that scales a vector in direction two by an input amount
            static NumericalMatrix<double> scalingTwoTensor(double scalingTwo) {
                NumericalMatrix<double> scalingTensor(3, 3);
                scalingTensor.setElement(0, 0, 1);
                scalingTensor.setElement(1, 1, scalingTwo);
                scalingTensor.setElement(2, 2, 1);
                return scalingTensor;
            }

            // The transformation tensor that scales a vector in direction three by an input amount
            static NumericalMatrix<double> scalingThreeTensor(double scalingThree) {
                NumericalMatrix<double> scalingTensor(3, 3);
                scalingTensor.setElement(0, 0, 1);
                scalingTensor.setElement(1, 1, 1);
                scalingTensor.setElement(2, 2, scalingThree);
                return scalingTensor;
            }

            // The transformation tensor that rotates a vector around axis 1 in the 2-3 plane by an input angle
            static NumericalMatrix<double> rotationTensorOne(double angleInDegrees) {
                auto angleInRadians = Utility::Calculators::degreesToRadians(angleInDegrees);
                NumericalMatrix<double> rotationTensorOne(3, 3);
                rotationTensorOne.setElement(0, 0, 1);
                rotationTensorOne.setElement(1, 1, cos(angleInRadians));
                rotationTensorOne.setElement(1, 2, -sin(angleInRadians));
                rotationTensorOne.setElement(2, 1, sin(angleInRadians));
                rotationTensorOne.setElement(2, 2, cos(angleInRadians));
                return rotationTensorOne;
            }

            // The transformation tensor that rotates a vector around axis 2 in the 1-3 plane by an input angle
            static NumericalMatrix<double> rotationTensorTwo(double angleInDegrees) {
                auto angleInRadians = Utility::Calculators::degreesToRadians(angleInDegrees);
                NumericalMatrix<double> rotationTensorTwo(3, 3);
                rotationTensorTwo.setElement(0, 0, cos(angleInRadians));
                rotationTensorTwo.setElement(0, 2, sin(angleInRadians));
                rotationTensorTwo.setElement(1, 1, 1);
                rotationTensorTwo.setElement(2, 0, -sin(angleInRadians));
                rotationTensorTwo.setElement(2, 2, cos(angleInRadians));
                return rotationTensorTwo;
            }

            // The transformation tensor that rotates a vector around axis 3 in the 1-2 plane by an input angle
            static NumericalMatrix<double> rotationTensorThree(double angleInDegrees) {
                auto angleInRadians = Utility::Calculators::degreesToRadians(angleInDegrees);
                NumericalMatrix<double> rotationTensorThree(3, 3);
                rotationTensorThree.setElement(0, 0, cos(angleInRadians));
                rotationTensorThree.setElement(0, 1, -sin(angleInRadians));
                rotationTensorThree.setElement(1, 0, sin(angleInRadians));
                rotationTensorThree.setElement(1, 1, cos(angleInRadians));
                rotationTensorThree.setElement(2, 2, 1);
                return rotationTensorThree;
            }

            // The transformation tensor that shears a vector in direction 1 in the 1-2 plane by an input angle
            static NumericalMatrix<double> shearTensorOne_planeOneTwo(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                NumericalMatrix<double> shearTensorOne_planeOneTwo(3, 3);
                shearTensorOne_planeOneTwo.setElement(0, 0, 1);
                shearTensorOne_planeOneTwo.setElement(0, 1, ::tan(shearAngleInDegrees));
                shearTensorOne_planeOneTwo.setElement(1, 1, 1);
                shearTensorOne_planeOneTwo.setElement(2, 2, 1);
                return shearTensorOne_planeOneTwo;
            }

            // The transformation tensor that shears a vector in direction 2 in the 1-2 plane by an input angle
            static NumericalMatrix<double> shearTensorTwo_planeOneTwo(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                NumericalMatrix<double> shearTensorTwo_planeOneTwo(3, 3);
                shearTensorTwo_planeOneTwo.setElement(0, 0, 1);
                shearTensorTwo_planeOneTwo.setElement(1, 0, ::tan(shearAngleInDegrees));
                shearTensorTwo_planeOneTwo.setElement(1, 1, 1);
                shearTensorTwo_planeOneTwo.setElement(2, 2, 1);
                return shearTensorTwo_planeOneTwo;
            }

            // The transformation tensor that shears a vector in direction 1 in the 1-3 plane by an input angle
            static NumericalMatrix<double> shearTensorOne_planeOneThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                NumericalMatrix<double> shearTensorOne_planeOneThree(3, 3);
                shearTensorOne_planeOneThree.setElement(0, 0, 1);
                shearTensorOne_planeOneThree.setElement(0, 2, ::tan(shearAngleInDegrees));
                shearTensorOne_planeOneThree.setElement(1, 1, 1);
                shearTensorOne_planeOneThree.setElement(2, 2, 1);
                return shearTensorOne_planeOneThree;
            }

            // The transformation tensor that shears a vector in direction 3 in the 1-3 plane by an input angle
            static NumericalMatrix<double> shearTensorThree_planeOneThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                NumericalMatrix<double> shearTensorThree_planeOneThree(3, 3);
                shearTensorThree_planeOneThree.setElement(0, 0, 1);
                shearTensorThree_planeOneThree.setElement(2, 0, ::tan(shearAngleInDegrees));
                shearTensorThree_planeOneThree.setElement(1, 1, 1);
                shearTensorThree_planeOneThree.setElement(2, 2, 1);
                return shearTensorThree_planeOneThree;
            }

            // The transformation tensor that shears a vector in direction 2 in the 2-3 plane by an input angle
            static NumericalMatrix<double> shearTensorTwo_planeTwoThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                NumericalMatrix<double> shearTensorTwo_planeTwoThree(3, 3);
                shearTensorTwo_planeTwoThree.setElement(0, 0, 1);
                shearTensorTwo_planeTwoThree.setElement(1, 1, 1);
                shearTensorTwo_planeTwoThree.setElement(1, 2, ::tan(shearAngleInDegrees));
                shearTensorTwo_planeTwoThree.setElement(2, 2, 1);
                return shearTensorTwo_planeTwoThree;
            }

            // The transformation tensor that shears a vector in direction 3 in the 2-3 plane by an input angle
            static static NumericalMatrix<double> shearTensorThree_planeTwoThree(double shearAngleInDegrees) {
                auto shearAngleInRadians = Utility::Calculators::degreesToRadians(shearAngleInDegrees);
                NumericalMatrix<double> shearTensorThree_planeTwoThree(3, 3);
                shearTensorThree_planeTwoThree.setElement(0, 0, 1);
                shearTensorThree_planeTwoThree.setElement(1, 1, 1);
                shearTensorThree_planeTwoThree.setElement(2, 1, ::tan(shearAngleInDegrees));
                shearTensorThree_planeTwoThree.setElement(2, 2, 1);
                return shearTensorThree_planeTwoThree;
            }

            // The transformation tensor that reflects about  axis 1  in the 1-2 plane
            static NumericalMatrix<double> ReflectionTensorOne_planeOneTwo() {
                NumericalMatrix<double> reflectionTensorOne_planeOneTwo(3, 3);
                reflectionTensorOne_planeOneTwo.setElement(0, 0, 1);
                reflectionTensorOne_planeOneTwo.setElement(1, 1, -1);
                reflectionTensorOne_planeOneTwo.setElement(2, 2, 1);
                return reflectionTensorOne_planeOneTwo;
            }

            // The transformation tensor that reflects about  axis 2  in the 1-2 plane
            static NumericalMatrix<double> ReflectionTensorTwo_planeOneTwo() {
                NumericalMatrix<double> reflectionTensorTwo_planeOneTwo(3, 3);
                reflectionTensorTwo_planeOneTwo.setElement(0, 0, -1);
                reflectionTensorTwo_planeOneTwo.setElement(1, 1, 1);
                reflectionTensorTwo_planeOneTwo.setElement(2, 2, 1);
                return reflectionTensorTwo_planeOneTwo;
            }

            // The transformation tensor that reflects about  axis 1  in the 1-3 plane
            static NumericalMatrix<double> ReflectionTensorOne_planeOneThree() {
                NumericalMatrix<double> reflectionTensorOne_planeOneThree(3, 3);
                reflectionTensorOne_planeOneThree.setElement(0, 0, 1);
                reflectionTensorOne_planeOneThree.setElement(1, 1, 1);
                reflectionTensorOne_planeOneThree.setElement(2, 2, -1);
                return reflectionTensorOne_planeOneThree;
            }

            // The transformation tensor that reflects about  axis 3  in the 1-3 plane
            static NumericalMatrix<double> ReflectionTensorThree_planeOneThree() {
                NumericalMatrix<double> reflectionTensorThree_planeOneThree(3, 3);
                reflectionTensorThree_planeOneThree.setElement(0, 0, -1);
                reflectionTensorThree_planeOneThree.setElement(1, 1, 1);
                reflectionTensorThree_planeOneThree.setElement(2, 2, 1);
                return reflectionTensorThree_planeOneThree;
            }

            // The transformation tensor that reflects about  axis 2  in the 2-3 plane
            static NumericalMatrix<double> ReflectionTensorTwo_planeTwoThree() {
                NumericalMatrix<double> reflectionTensorTwo_planeTwoThree(3, 3);
                reflectionTensorTwo_planeTwoThree.setElement(0, 0, 1);
                reflectionTensorTwo_planeTwoThree.setElement(1, 1, 1);
                reflectionTensorTwo_planeTwoThree.setElement(2, 2, -1);
                return reflectionTensorTwo_planeTwoThree;
            }

            // The transformation tensor that reflects about  axis 3  in the 2-3 plane
            static NumericalMatrix<double> ReflectionTensorThree_planeTwoThree() {
                NumericalMatrix<double> reflectionTensorThree_planeTwoThree(3, 3);
                reflectionTensorThree_planeTwoThree.setElement(0, 0, 1);
                reflectionTensorThree_planeTwoThree.setElement(1, 1, -1);
                reflectionTensorThree_planeTwoThree.setElement(2, 2, 1);
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

/*static NumericalVector<double> Shear(NumericalVector<double> &array, double shearAngleInDegrees, PositioningInSpace::Direction axis){
    NumericalVector<double> shearedArray;
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