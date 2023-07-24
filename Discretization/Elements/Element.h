//
// Created by hal9000 on 7/18/23.
//

#ifndef UNTITLED_ELEMENT_H
#define UNTITLED_ELEMENT_H

#include "../Node/Node.h"

namespace Discretization {
    
    enum ElementType {
        Line,
        Triangle,
        Quadrilateral,
        Wedge,
        Hexahedron
    };

    class Element {
    public:
        Element(unsigned id, vector<Node*> nodes, ElementType type);
        
        Element& operator=(const Element& other);
        
        const unique_ptr<vector<Node*>> &nodes() const;
        
        unique_ptr<list<DegreeOfFreedom*>> degreesOfFreedom() const;
        
        unsigned id() const;
        
        const ElementType &type() const;
        
        void deallocate();
        
        void printElement(bool printCoordinates = false);
        
        unsigned numberOfNodes() const;
        
        unsigned numberOfDegreesOfFreedom() const;

    private:
        unique_ptr<vector<Node*>> _nodes;
        unique_ptr<unsigned> _id;
        ElementType _type;
    };

} // Discretization

#endif //UNTITLED_ELEMENT_H
