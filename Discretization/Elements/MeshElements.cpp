//
// Created by hal9000 on 7/18/23.
//

#include "MeshElements.h"

namespace Discretization {
    
        MeshElements::MeshElements(unique_ptr<vector<Element*>> elements, ElementType type) {
            _elements = std::move(elements);
            _type = type;
        }
        
        Element *MeshElements::getElement(unsigned int id) {
            return _elements->at(id);
        }
        
        unsigned MeshElements::numberOfElements() const {
            return _elements->size();
        }
        
        void MeshElements::deallocate() {
            for (auto &element : *_elements) {
                element->deallocate();
            }
            _elements->clear();
            _elements.reset();
        }
        
        const ElementType &MeshElements::elementType() const {
            return _type;
        }
        
} // Discretization