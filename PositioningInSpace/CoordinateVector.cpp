//
// Created by hal9000 on 1/8/23.
//

#include "CoordinateVector.h"

namespace PositioningInSpace{
    vector<double> CoordinateVector::getCoordinateVectorInEntity(PhysicalSpaceEntities &thisPhysicalSpace, PhysicalSpaceEntities &physicalSpace) {
        switch (thisPhysicalSpace) {
            
            case (PhysicalSpaceEntities::One_axis):
                switch (physicalSpace) {
                    case (PhysicalSpaceEntities::One_axis):
                        return _positionVector;
                    case (PhysicalSpaceEntities::Two_axis): 
                        return {0};
                    case (PhysicalSpaceEntities::Three_axis):
                        return {0};
                    case (PhysicalSpaceEntities::OneTwoThree_volume):
                        return {0,0};
                    default:
                        return {0};
                    case OneTwo_plane:
                        break;
                    case OneThree_plane:
                        break;
                    case TwoThree_plane:
                        break;
                    case NullSpace:
                        break;
                }
                break;
            
            case (PhysicalSpaceEntities::Two_axis):
                switch (physicalSpace) {
                    case (PhysicalSpaceEntities::Two_axis):
                        return _positionVector;
                        
                    default:
                        return {0};
                }
                break;
            
            case (PhysicalSpaceEntities::Three_axis):
                switch (physicalSpace) {
                    case (PhysicalSpaceEntities::Three_axis):
                        return _positionVector;
                    default:
                        return {0};
                }
                break;
            
        }
        
        
    }
    
    vector<double> CoordinateVector::getCoordinateVectorIn3D(PhysicalSpaceEntities &physicalSpace) {
        
    }
    
    void CoordinateVector::setCoordinateVector(vector<double> coordinateVector, PhysicalSpaceEntities &physicalSpace) {
        
    }
    
    unsigned CoordinateVector::dimensions() {
        return _positionVector.size();
    }
    
    
} // PositioningInSpace