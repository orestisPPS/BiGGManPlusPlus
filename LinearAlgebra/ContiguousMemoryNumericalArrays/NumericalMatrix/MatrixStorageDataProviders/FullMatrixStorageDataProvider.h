//
// Created by hal9000 on 9/2/23.
//

#ifndef UNTITLED_FULLMATRIXSTORAGEDATAPROVIDER_H
#define UNTITLED_FULLMATRIXSTORAGEDATAPROVIDER_H


#include "NumericalMatrixStorageDataProvider.h"

namespace LinearAlgebra {

    template<typename T>
    class FullMatrixStorageDataProvider : public NumericalMatrixStorageDataProvider<T> {
        
    public:
        explicit FullMatrixStorageDataProvider(unsigned numberOfRows, unsigned numberOfColumns, unsigned availableThreads) :
                NumericalMatrixStorageDataProvider<T>(numberOfRows, numberOfColumns, availableThreads) {
            this->_values = make_shared<NumericalVector<T>>(numberOfRows * numberOfColumns, 0, availableThreads);
            this->_storageType = NumericalMatrixStorageType::FullMatrix;
        }
        
        T& getElement(unsigned row, unsigned column) override {
            return this->_values->getDataPointer()[row * this->_numberOfColumns + column];
        }
        
        void setElement(unsigned row, unsigned column, T value) override {
            this->_values->getDataPointer()[row * this->_numberOfColumns + column] = value;
        }
        
        void eraseElement(unsigned row, unsigned column) override {
            this->_values->getDataPointer()[row * this->_numberOfColumns + column] = static_cast<T>(0);
        }

        vector<shared_ptr<NumericalVector<unsigned>>> getSupplementaryVectors() override{
            throw runtime_error("FullMatrixStorage does not have supplementary vectors.");
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

        shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) override {
            if (row >= this->_numberOfRows){
                throw runtime_error("Row index out of bounds.");
            }
            auto rowVector = make_shared<NumericalVector<T>>(this->_numberOfColumns);
            auto vectorData = rowVector->getDataPointer();
            auto thisData = this->_values->getDataPointer();
            for (unsigned int column = 0; column < this->_numberOfColumns; column++){
                vectorData[column] = thisData[row * this->_numberOfColumns + column];
            }
            return rowVector;
        }
        
        void getRow(unsigned row, shared_ptr<NumericalVector<T>> &vector) override {
            if (row >= this->_numberOfRows){
                throw runtime_error("Row index out of bounds.");
            }
            auto vectorData = vector->getDataPointer();
            auto thisData = this->_values->getDataPointer();
            for (unsigned int column = 0; column < this->_numberOfColumns; column++){
                vectorData[column] = thisData[row * this->_numberOfColumns + column];
            }
        }
        
        shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) override {
            if (column >= this->_numberOfColumns){
                throw runtime_error("Column index out of bounds.");
            }
            auto columnVector = make_shared<NumericalVector<T>>(this->_numberOfRows);
            auto vectorData = columnVector->getDataPointer();
            auto thisData = this->_values->getDataPointer();
            for (unsigned int row = 0; row < this->_numberOfRows; row++){
                vectorData[row] = thisData[row * this->_numberOfColumns + column];
            }
            return columnVector;
        }
        
        void getColumn(unsigned column, shared_ptr<NumericalVector<T>> &vector) override {
            if (column >= this->_numberOfColumns){
                throw runtime_error("Column index out of bounds.");
            }
            auto vectorData = vector->getDataPointer();
            auto thisData = this->_values->getDataPointer();
            for (unsigned int row = 0; row < this->_numberOfRows; row++){
                vectorData[row] = thisData[row * this->_numberOfColumns + column];
            }
        }

        void deepCopy(NumericalMatrixStorageDataProvider<T> &inputMatrixData) override {
            if (inputMatrixData.getStorageType() != this->_storageType)
                throw runtime_error("Cannot copy from a different storage type.");

            auto inputValues = inputMatrixData.getValues();

            this->_values = make_shared<NumericalVector<T>>(*inputValues);
        }

        bool areElementsEqual(NumericalMatrixStorageDataProvider<T> &inputMatrixData) override {
            if (inputMatrixData.getStorageType() != this->_storageType)
                throw runtime_error("Cannot compare with a different storage type.");

            auto inputValues = inputMatrixData.getValues();

            return ((this->_values == inputValues));
        }
        
        
    };

} // LinearAlgebra

#endif //UNTITLED_FULLMATRIXSTORAGEDATAPROVIDER_H
