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
    protected:
        map<vector<shared_ptr<unsigned>>, shared_ptr<T>> _coordinateMap;
}

#endif //UNTITLED_SPARSEMATRIXSTORAGE_H
