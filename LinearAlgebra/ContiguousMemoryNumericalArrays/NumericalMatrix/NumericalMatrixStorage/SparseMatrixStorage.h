//
// Created by hal9000 on 9/3/23.
//

#ifndef UNTITLED_SPARSEMATRIXSTORAGE_H
#define UNTITLED_SPARSEMATRIXSTORAGE_H

#include <map>
#include "NumericalMatrixStorage.h"

namespace LinearAlgebra{
    template <typename T>
    class SparseMatrixStorage : public NumericalMatrixStorage<T> {
    public:
        SparseMatrixStorage(unsigned &numberOfRows, unsigned &numberOfColumns
    protected:
        /**
        * @brief A list storing the matrix elements and their indices.
        *
        * This list contains tuples that represent non-zero matrix elements.
        * Each tuple consists of:
        * 1. A shared pointer to the row index of the element.
        * 2. A shared pointer to the column index of the element.
        * 3. A shared pointer to the value of the element.
        *
        * This list can be sorted in a row-major order using the `_sortElements()` method.
        */
        list<tuple<shared_ptr<unsigned int>, shared_ptr<unsigned int>, shared_ptr<T>>> _elements;

        /**
         * @brief Sorts the matrix elements in row-major order.
         *
         * This function sorts the `_elements` list to ensure that the elements are
         * ordered based on their row and column indices in a row-major fashion.
         * 
         * Row-major order means that elements are sorted primarily by their row indices.
         * If two elements have the same row index, then they are sorted by their column indices.
         */
        void _sortElements(){
            _elements.sort([](const tuple<shared_ptr<unsigned int>, shared_ptr<unsigned int>, shared_ptr<T>> &a,
                              const tuple<shared_ptr<unsigned int>, shared_ptr<unsigned int>, shared_ptr<T>> &b) -> bool {
                return *get<0>(a) < *get<0>(b) || (*get<0>(a) == *get<0>(b) && *get<1>(a) < *get<1>(b));
            });
        }
}

#endif //UNTITLED_SPARSEMATRIXSTORAGE_H
