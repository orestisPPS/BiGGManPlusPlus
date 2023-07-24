//
// Created by hal9000 on 7/18/23.
//

#include "Element.h"

namespace Discretization {
    Element::Element(unsigned id, vector<Node*> nodes, ElementType type) {
        _id = make_unique<unsigned>(id);
        _nodes = make_unique<vector<Node*>>(std::move(nodes));
        _type = type;
    }
    
    Element& Element::operator=(const Element& other) {
        if (this != &other) {
            *_id = *other._id;
            *_nodes = *other._nodes;
            _type = other._type;
        }
        return *this;
    }
    
    const unique_ptr<vector<Node*>> &Element::nodes() const {
        return _nodes;
    }
    
    unique_ptr<list<DegreeOfFreedom*>> Element::degreesOfFreedom() const {
     auto dofs = make_unique<list<DegreeOfFreedom*>>();
     for (auto &node : *_nodes) {
         for (auto &dof : *node->degreesOfFreedom) {
             dofs->push_back(dof);
         }
     }
        return dofs;
    }
    
    unsigned Element::id() const {
        return *_id;
    }
    
    const ElementType &Element::type() const {
        return _type;
    }
    
    void Element::deallocate() {
        _nodes->clear();
        _nodes.reset();
        _id.reset();
    }
    
    void Element::printElement(bool printCoordinates) {
        cout << "Element: " << (*_id) << endl;
        cout << "Nodes: " << endl;
        cout << "---------------------------------------------------------" << endl;
        for (auto &node : *_nodes) {
            cout << "Node: " << (*node->id.global) << endl;
            if (printCoordinates) {
                node->printNode();
            }
        }
        cout << "---------------------------------------------------------" << endl;
    }
    
    unsigned Element::numberOfNodes() const {
        return _nodes->size();
    }
    
    unsigned Element::numberOfDegreesOfFreedom() const {
        return degreesOfFreedom()->size();
    }
} // Discretization