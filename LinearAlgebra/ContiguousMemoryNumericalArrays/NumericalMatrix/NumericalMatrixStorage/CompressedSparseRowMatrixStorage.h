//
// Created by hal9000 on 9/2/23.
//

#ifndef UNTITLED_COMPRESSEDSPARSEROWMATRIXSTORAGE_H
#define UNTITLED_COMPRESSEDSPARSEROWMATRIXSTORAGE_H

#include "NumericalMatrixStorage.h"

namespace LinearAlgebra {
    template <typename T>
    class CompressedSparseRowMatrixStorage : public NumericalMatrixStorage<T> {
    public:
        explicit CompressedSparseRowMatrixStorage(unsigned &numberOfRows, unsigned &numberOfColumns) :
                NumericalMatrixStorage<T>(CSR, numberOfRows, numberOfColumns) {
                auto numberOfNonZeroElements = static_cast<unsigned int>(this->_numberOfRows * this->_numberOfColumns * this->_sparsityPercentage);
                this->_values = make_shared<NumericalVector<T>>(numberOfNonZeroElements);
                this->_columnIndices = make_shared<NumericalVector<unsigned>>(numberOfNonZeroElements);
                this->_rowPointers = make_shared<NumericalVector<unsigned>>(this->_numberOfRows + 1);
        }

        const T &operator()(unsigned int row, unsigned int column) const override {
            unsigned int rowStart = this->_rowPointers->operator[](row);
            unsigned int rowEnd = this->_rowPointers->operator[](row + 1);
            for (unsigned int i = rowStart; i < rowEnd; i++) {
                if (this->_columnIndices->operator[](i) == column) {
                    return this->_values->operator[](i);
                }
            }
            return 0.0;
        }

        void initializeElementAssignment() override {
            if (this->_elementAssignmentRunning) {
                throw runtime_error("Element assignment is already running. Call finalizeElementAssignment() first.");
            }
            this->_elementAssignmentRunning = true;
            _values->fill(0.0);
            _columnIndices->fill(0);
            _rowPointers->fill(0);
        }

        void finalizeElementAssignment() override {
            if (!this->_elementAssignmentRunning) {
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }
            auto lastNonZeroElementIndex = 0;
            auto data = this->_values->getDataPointer();
            for (auto &value : *data) {
                if (value != 0.0) {
                    lastNonZeroElementIndex++;
                }
                else {
                    break;
                }
            }
            this->_values->resize(lastNonZeroElementIndex);
            this->_columnIndices->resize(lastNonZeroElementIndex);
        }

        void setElement(unsigned int row, unsigned int column, const T &value) override {
            if (!this->_elementAssignmentRunning) {
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }
            unsigned int rowStart = (*_rowPointers)[row];
            unsigned int rowEnd = (*_rowPointers)[row + 1];
            for (unsigned int i = rowStart; i < rowEnd; i++) {
                if ((*_columnIndices)[i] == column) {
                    (*_values)[i] = value;
                    return;
                }
            }
            this->_values->operator[](rowEnd) = value;
            this->_columnIndices->operator[](row + 1) = column;
            this->_rowPointers->operator[](row + 1) = rowEnd + 1;
        }
        
        
        private:
            shared_ptr<NumericalVector<T>> _values;
            shared_ptr<NumericalVector<unsigned>> _columnIndices;
            shared_ptr<NumericalVector<unsigned>> _rowPointers;

    };

} // LinearAlgebra

#endif //UNTITLED_COMPRESSEDSPARSEROWMATRIXSTORAGE_H
