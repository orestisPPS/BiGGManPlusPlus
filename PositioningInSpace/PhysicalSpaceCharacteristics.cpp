//
// Created by hal9000 on 1/5/23.
//

#include "SpaceCharacteristics.h"

namespace PositioningInSpace {
    
        SpaceCharacteristics::SpaceCharacteristics(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection,
                                                   CoordinateSystem coordinateSystem) {
            CheckInput(nodesPerDirection);
            this->coordinateSystem = coordinateSystem;
            CalculateDimensions();
            CalculatePhysicalSpace(nodesPerDirection);
        }
        
        void SpaceCharacteristics::CheckInput(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection) {
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
        
        void SpaceCharacteristics::CalculateDimensions() {
            if (physicalSpace == PhysicalSpace::One_axis || physicalSpace == PhysicalSpace::Two_axis || physicalSpace == PhysicalSpace::Three_axis)
                Dimensions = 1;
            else if (physicalSpace == PhysicalSpace::OneTwo_plane || physicalSpace == PhysicalSpace::OneThree_plane || physicalSpace == PhysicalSpace::TwoThree_plane)
                Dimensions = 2;
            else if (physicalSpace == PhysicalSpace::OneTwoThree_volume)
                Dimensions = 3;
            else
                throw std::invalid_argument("I'm sorry Dave, I'm afraid I can't do that");
            }
    
        void SpaceCharacteristics::CalculatePhysicalSpace(std::map<PositioningInSpace::Direction, unsigned > &nodesPerDirection) {
            auto nn1 = nodesPerDirection.at(Direction::One);
            auto nn2 = nodesPerDirection.at(Direction::Two);
            auto nn3 = nodesPerDirection.at(Direction::Three);

            //1D
            if (nn1 > 1 && nn2 == 1 && nn3 == 1)
                physicalSpace = PhysicalSpace::One_axis;
            else if (nn1 == 1 && nn2 > 1 && nn3 == 1)
                physicalSpace = PhysicalSpace::Two_axis;
            else if (nn1 == 1 && nn2 == 1 && nn3 > 1)
                physicalSpace = PhysicalSpace::Three_axis;
            //2D
            else if (nn1 > 1 && nn2 > 1 && nn3 == 1)
                physicalSpace = PhysicalSpace::OneTwo_plane;
            else if (nn1 > 1 && nn2 == 1 && nn3 > 1)
                physicalSpace = PhysicalSpace::OneThree_plane;
            else if (nn1 == 1 && nn2 > 1 && nn3 > 1)
                physicalSpace = PhysicalSpace::TwoThree_plane;
            //3D
            else if (nn1 > 1 && nn2 > 1 && nn3 > 1)
                physicalSpace = PhysicalSpace::OneTwoThree_volume;
            else
                throw std::invalid_argument("I'm sorry Dave, I'm afraid I can't do that");
        }
    
    }; // PositioningInSpace
