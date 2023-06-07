//
// Created by hal9000 on 3/10/23.
//

#ifndef UNTITLED_ENUMERABLESTRANFORMATIONS_H
#define UNTITLED_ENUMERABLESTRANFORMATIONS_H

#include <list>
#include <vector>
using namespace std;
namespace Utility {
    
        template<typename T>
        static vector<T> *linkedListToVector(list<T> *list) {
            auto *vector = new std::vector<T>(list->size());
            for (auto item : *list) {
                vector->push_back(item);
            }
            return vector;
        }

        template<typename T>
        static std::list<T> *vectorToLinkedList(vector<T> *vector) {
            auto *list = new std::list<T>();
            for (auto item : *vector) {
                list->push_back(item);
            }
            return list;
        }

} // Utility

#endif //UNTITLED_ENUMERABLESTRANFORMATIONS_H
