//
// Created by hal9000 on 1/8/23.
//

#ifndef UNTITLED_LINE_H
#define UNTITLED_LINE_H

#include "PhysicalSpaceEntity.h"

namespace PositioningInSpace {

    class Line : public PhysicalSpaceEntity {
    public:
        explicit Line(PhysicalSpaceEntities lineType);
        const PhysicalSpaceEntities  &type() override;
    private:
        PhysicalSpaceEntities _type ;
        static bool _checkInput(PhysicalSpaceEntities type);
    };

} // PositioningInSpace

#endif //UNTITLED_LINE_H
