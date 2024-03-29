//
// Created by hal9000 on 7/18/23.
//

#ifndef UNTITLED_MESHELEMENTS_H
#define UNTITLED_MESHELEMENTS_H

#include "../Node/Node.h"
#include "Element.h"

namespace Discretization {

    class MeshElements {
    public:
        MeshElements(unique_ptr<vector<Element*>> elements, ElementType type);
        
        //->at() returns a reference to the element
        Element* getElement(unsigned id);
        
        unsigned numberOfElements() const;
        
        void deallocate();
        
        const ElementType &elementType() const;
        
        
    private:
        
        ElementType _type;
        
        unique_ptr<vector<Element*>> _elements;
        
        unique_ptr<map<Direction, vector<Element*>>> _boundaryElements;
    };

} // Discretization

#endif //UNTITLED_MESHELEMENTS_H
