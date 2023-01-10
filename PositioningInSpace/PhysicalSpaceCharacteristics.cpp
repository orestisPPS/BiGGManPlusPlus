//
// Created by hal9000 on 1/5/23.
//

#include "PhysicalSpaceCharacteristics.h"
#include "CoordinateSystem.h"

namespace PositioningInSpace {
    
        PhysicalSpaceCharacteristics::PhysicalSpaceCharacteristics(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection,
                                                                   CoordinateSystem coordinateSystem) {
            CheckInput(nodesPerDirection);
            this->coordinateSystem = coordinateSystem;
            CalculateDimensions();
            CalculatePhysicalSpace(nodesPerDirection);
        }
        
        void PhysicalSpaceCharacteristics::CheckInput(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection) {
            auto nn1 = nodesPerDirection.at(Direction::One);
            auto nn2 = nodesPerDirection.at(Direction::Two);
            auto nn3 = nodesPerDirection.at(Direction::Three);
            if (nn1 == 0)
                nodesPerDirection.at(Direction::One) = 1;
            if (nn2 == 0)
                nodesPerDirection.at(Direction::Two) = 1;
            if (nn3 == 0)
                nodesPerDirection.at(Direction::Three) = 1;
            
            if (nn1 == 1 && nn2 == 1 && nn3 == 1)
                throw std::invalid_argument("At least one direction must have more than one node");
        }
        
        void PhysicalSpaceCharacteristics::CalculateDimensions() {
            if (physicalSpace == PhysicalSpaceEntities::One_axis || physicalSpace == PhysicalSpaceEntities::Two_axis || physicalSpace == PhysicalSpaceEntities::Three_axis)
                Dimensions = 1;
            else if (physicalSpace == PhysicalSpaceEntities::OneTwo_plane || physicalSpace == PhysicalSpaceEntities::OneThree_plane || physicalSpace == PhysicalSpaceEntities::TwoThree_plane)
                Dimensions = 2;
            else if (physicalSpace == PhysicalSpaceEntities::OneTwoThree_volume)
                Dimensions = 3;
            else
                throw std::invalid_argument("I'm sorry Dave, I'm afraid I can't do that");
            }
    
        void PhysicalSpaceCharacteristics::CalculatePhysicalSpace(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection) {
            auto nn1 = nodesPerDirection.at(Direction::One);
            auto nn2 = nodesPerDirection.at(Direction::Two);
            auto nn3 = nodesPerDirection.at(Direction::Three);

            //1D
            if (nn1 > 1 && nn2 == 1 && nn3 == 1)
                physicalSpace = PhysicalSpaceEntities::One_axis;
            else if (nn1 == 1 && nn2 > 1 && nn3 == 1)
                physicalSpace = PhysicalSpaceEntities::Two_axis;
            else if (nn1 == 1 && nn2 == 1 && nn3 > 1)
                physicalSpace = PhysicalSpaceEntities::Three_axis;
            //2D
            else if (nn1 > 1 && nn2 > 1 && nn3 == 1)
                physicalSpace = PhysicalSpaceEntities::OneTwo_plane;
            else if (nn1 > 1 && nn2 == 1 && nn3 > 1)
                physicalSpace = PhysicalSpaceEntities::OneThree_plane;
            else if (nn1 == 1 && nn2 > 1 && nn3 > 1)
                physicalSpace = PhysicalSpaceEntities::TwoThree_plane;
            //3D
            else if (nn1 > 1 && nn2 > 1 && nn3 > 1)
                physicalSpace = PhysicalSpaceEntities::OneTwoThree_volume;
            else
                throw std::invalid_argument("I'm sorry Dave, I'm afraid I can't do that");
        }
    
    }; // PositioningInSpace
