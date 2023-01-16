//
// Created by hal9000 on 1/8/23.
//

#include "CoordinateVector.h"

namespace PositioningInSpace{
    vector<double> CoordinateVector::getCoordinateVectorInEntity(const PhysicalSpaceEntities &thisPhysicalSpace,
                                                                PhysicalSpaceEntities physicalSpace) {
        switch (thisPhysicalSpace) {

            case OneTwoThree_volume:
                if (physicalSpace == OneTwoThree_volume)
                    return _positionVector;
                else if (physicalSpace == OneTwo_plane)
                    return {_positionVector[0], _positionVector[1]};
                else if (physicalSpace == OneThree_plane)
                    return {_positionVector[0], _positionVector[2]};
                else if (physicalSpace == TwoThree_plane)
                    return {_positionVector[1], _positionVector[2]};
                else if (physicalSpace == One_axis)
                    return {_positionVector[0]};
                else if (physicalSpace == Two_axis)
                    return {_positionVector[1]};
                else if (physicalSpace == Three_axis)
                    return {_positionVector[2]};

                else
                    return {};

            case OneTwo_plane:
                if (physicalSpace == OneTwo_plane)
                    return _positionVector;
                else if (physicalSpace == OneThree_plane)
                    return {0, 0};
                else if (physicalSpace == TwoThree_plane)
                    return {_positionVector[1], 0};
                else if (physicalSpace == OneTwoThree_volume)
                    return {_positionVector[0], _positionVector[1], 0};
                if (physicalSpace == One_axis)
                    return {_positionVector[0]};
                else if (physicalSpace == Two_axis)
                    return {_positionVector[1]};
                else if (physicalSpace == Three_axis)
                    return {0};
                else
                    return {};

            case OneThree_plane:
                if (physicalSpace == OneThree_plane)
                    return _positionVector;
                else if (physicalSpace == OneTwo_plane)
                    return {_positionVector[0], 0};
                else if (physicalSpace == TwoThree_plane)
                    return {0, _positionVector[1]};
                else if (physicalSpace == OneTwoThree_volume)
                    return {_positionVector[0], 0, _positionVector[1]};
                else if (physicalSpace == One_axis)
                    return {_positionVector[0]};
                else if (physicalSpace == Two_axis)
                    return {0};
                else if (physicalSpace == Three_axis)
                    return {_positionVector[1]};
                else
                    return {};

            case TwoThree_plane:
                if (physicalSpace == TwoThree_plane)
                    return _positionVector;
                else if (physicalSpace == OneTwo_plane)
                    return {0, _positionVector[0]};
                else if (physicalSpace == OneThree_plane)
                    return {0, _positionVector[1]};
                else if (physicalSpace == One_axis)
                    return {0};
                else if (physicalSpace == Two_axis)
                    return {_positionVector[0]};
                else if (physicalSpace == Three_axis)
                    return {_positionVector[1]};
                else if (physicalSpace == OneTwoThree_volume)
                    return {0, _positionVector[0], _positionVector[1]};
                else
                    return {};

            case One_axis:
                if (physicalSpace == One_axis)
                    return _positionVector;
                else if (physicalSpace == Two_axis || physicalSpace == Three_axis)
                    return {0};
                else if (physicalSpace == OneTwo_plane || physicalSpace == OneThree_plane)
                    return {_positionVector[0], 0};
                else if (physicalSpace == TwoThree_plane)
                    return {0, 0};
                else if (physicalSpace == OneTwoThree_volume)
                    return {_positionVector[0], 0, 0};
                else
                    return {};

            case Two_axis:
                if (physicalSpace == One_axis || physicalSpace == Three_axis)
                    return {0};
                else if (physicalSpace == Two_axis)
                    return _positionVector;
                else if (physicalSpace == OneTwo_plane)
                    return {0, _positionVector[0]};
                else if (physicalSpace == OneThree_plane)
                    return {0, 0};
                else if (physicalSpace == TwoThree_plane)
                    return {_positionVector[0], 0};
                else if (physicalSpace == OneTwoThree_volume)
                    return {0, _positionVector[0], 0};
                else
                    return {};

            case Three_axis:
                if (physicalSpace == One_axis || physicalSpace == Two_axis)
                    return {0};
                else if (physicalSpace == Three_axis)
                    return _positionVector;
                else if (physicalSpace == OneTwo_plane)
                    return {0, 0};
                else if (physicalSpace == OneThree_plane)
                    return {0, _positionVector[0]};
                else if (physicalSpace == TwoThree_plane)
                    return {_positionVector[0], 0};
                else if (physicalSpace == OneTwoThree_volume)
                    return {0, 0, _positionVector[0]};
                else
                    return {};
            default:
                return {};
        }
    }
    
    vector<double> CoordinateVector::getCoordinateVectorIn3D(const PhysicalSpaceEntities &physicalSpace) {
        return getCoordinateVectorInEntity(physicalSpace, OneTwoThree_volume);
    }
    
    void CoordinateVector::setCoordinateVector(vector<double> coordinateVector, PhysicalSpaceEntities &physicalSpace) {
        if ((physicalSpace == OneTwoThree_volume && coordinateVector.size() == 3) ||
            (physicalSpace == OneTwo_plane || physicalSpace == TwoThree_plane || physicalSpace == OneThree_plane && coordinateVector.size() == 2) ||
            (physicalSpace == One_axis || physicalSpace == Two_axis || physicalSpace == Three_axis && coordinateVector.size() == 1))
            _positionVector = coordinateVector;
        else
            throw invalid_argument("Input position vector does not match the physical space dimensions.");
    }
    
    unsigned CoordinateVector::dimensions() {
        return _positionVector.size();
    }
    
} // PositioningInSpace