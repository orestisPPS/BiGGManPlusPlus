//
// Created by hal9000 on 9/2/23.
//

#ifndef UNTITLED_ROWMAJORMATRIXSTORAGE_H
#define UNTITLED_ROWMAJORMATRIXSTORAGE_H


#include "NumericalMatrixStorage.h"

namespace LinearAlgebra {

    template<typename T>
    class RowMajorMatrixStorage : public NumericalMatrixStorage<T> {
        
    public:
        explicit RowMajorMatrixStorage(unsigned &numberOfRows, unsigned &numberOfColumns) : 
        NumericalMatrixStorage<T>(RowMajor, numberOfRows, numberOfColumns) {
            this->_values = make_shared<NumericalVector<T>>(numberOfRows * numberOfColumns);
        }
        
        const T& operator()(unsigned int row, unsigned int column) const override {
            return this->getValues()->operator[](row * this->numberOfColumns() + column);
        }
        
        void initializeElementAssignment() override {
            if (this->_elementAssignmentRunning){
                throw runtime_error("Element assignment is already running. Call finalizeElementAssignment() first.");
            }
            this->_elementAssignmentRunning = true;
            this->_values->fill(0.0);
        }
        
        void finalizeElementAssignment() override {
            if (!this->_elementAssignmentRunning){
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }
            this->_elementAssignmentRunning = false;
        }
        
        void setElement(unsigned int row, unsigned int column, const T &value) override {
            if (!this->_elementAssignmentRunning){
                throw runtime_error("Element assignment is not running. Call initializeElementAssignment() first.");
            }
            this->_values->operator[](row * this->_numberOfColumns + column) = value;
        }
        
        shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) override {
            if (row >= this->_numberOfRows){
                throw runtime_error("Row index out of bounds.");
            }
            auto rowVector = make_shared<NumericalVector<T>>(this->_numberOfColumns);
            for (unsigned int column = 0; column < this->_numberOfColumns; column++){
                rowVector->operator[](column) = this->operator()(row, column);
            }
            return rowVector;
        }
        
        void getRow(unsigned row, shared_ptr<NumericalVector<T>> &vector) override {
            if (row >= this->_numberOfRows){
                throw runtime_error("Row index out of bounds.");
            }
            for (unsigned int column = 0; column < this->_numberOfColumns; column++){
                vector->operator[](column) = this->operator()(row, column);
            }
        }
        
        shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) override {
            if (column >= this->_numberOfColumns){
                throw runtime_error("Column index out of bounds.");
            }
            auto columnVector = make_shared<NumericalVector<T>>(this->_numberOfRows);
            for (unsigned int row = 0; row < this->_numberOfRows; row++){
                columnVector->operator[](row) = this->operator()(row, column);
            }
            return columnVector;
        }
        
        void getColumn(unsigned column, shared_ptr<NumericalVector<T>> &vector) override {
            if (column >= this->_numberOfColumns){
                throw runtime_error("Column index out of bounds.");
            }
            for (unsigned int row = 0; row < this->_numberOfRows; row++){
                vector->operator[](row) = this->operator()(row, column);
            }
        }
        
    };

} // LinearAlgebra

#endif //UNTITLED_ROWMAJORMATRIXSTORAGE_H
