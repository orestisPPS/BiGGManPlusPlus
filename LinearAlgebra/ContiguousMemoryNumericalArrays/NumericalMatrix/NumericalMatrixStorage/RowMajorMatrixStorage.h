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
            this->_values->getDataPointer()[row * this->_numberOfColumns + column];
        }
        
        vector<vector<T>&> getNecessaryStorageVectors() override {
            return {*this->_values->getData()};
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

            this->_values->getDataPointer()[row * this->_numberOfColumns + column] = value;
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
        
        void matrixAdd(NumericalMatrixStorage<T> &inputMatrixData, NumericalMatrixStorage<T> &resultMatrixData, T scaleThis, T scaleOther) override {
            
            T *&thisMatrixDataPtr = this->_values->getDataPointer();
            T *&inputMatrixDataPtr = inputMatrixData[0]._values->getDataPointer();
            T *&resultMatrixDataPtr = resultMatrixData[0]._values->getDataPointer();
            
            auto matrixAddJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end; ++i) {
                    resultMatrixDataPtr[i] = scaleThis * thisMatrixDataPtr[i] +
                                             scaleOther * inputMatrixDataPtr[i];
                }
            };
            this->_threading.executeParallelJob(matrixAddJob, this->_values->size());
        }
        
        void matrixAddIntoThis(NumericalMatrixStorage<T> &inputMatrixData, T scaleThis, T scaleOther) override {
            
            T *&thisMatrixDataPtr = this->_values->getDataPointer();
            T *&inputMatrixDataPtr = inputMatrixData[0]._values->getDataPointer();
            
            auto matrixAddJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end; ++i) {
                    thisMatrixDataPtr[i] = scaleThis * thisMatrixDataPtr[i] +
                                           scaleOther * inputMatrixDataPtr[i];
                }
            };
            this->_threading.executeParallelJob(matrixAddJob, this->_values->size());
        }
        
        void matrixSubtract(NumericalMatrixStorage<T> &inputMatrixData, NumericalMatrixStorage<T> &resultMatrixData, T scaleThis, T scaleOther) override {

            T *&thisMatrixDataPtr = this->_values->getDataPointer();
            T *&inputMatrixDataPtr = inputMatrixData[0]._values->getDataPointer();
            T *&resultMatrixDataPtr = resultMatrixData[0]._values->getDataPointer();

            auto matrixSubtractJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end; ++i) {
                    resultMatrixDataPtr[i] = scaleThis * thisMatrixDataPtr[i] -
                                             scaleOther * inputMatrixDataPtr[i];
                }
            };
            this->_threading.executeParallelJob(matrixSubtractJob, this->_values->size());
        }
        
        void matrixSubtractIntoThis(NumericalMatrixStorage<T> &inputMatrixData, T scaleThis, T scaleOther) override {

            T *&thisMatrixDataPtr = this->_values->getDataPointer();
            T *&inputMatrixDataPtr = inputMatrixData[0]._values->getDataPointer();

            auto matrixSubtractJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end; ++i) {
                    thisMatrixDataPtr[i] = scaleThis * thisMatrixDataPtr[i] -
                                           scaleOther * inputMatrixDataPtr[i];
                }
            };
            this->_threading.executeParallelJob(matrixSubtractJob, this->_values->size());
        }

        void matrixMultiply(NumericalMatrixStorage<T> &inputMatrixData, NumericalMatrixStorage<T> &resultMatrixData, T scaleThis, T scaleOther) override {

            T *&thisMatrixDataPtr = this->_values->getDataPointer();
            T *&inputMatrixDataPtr = inputMatrixData[0]._values->getDataPointer();
            T *&resultMatrixDataPtr = resultMatrixData[0]._values->getDataPointer();

            auto matrixMultiplyJob = [&](unsigned startRow, unsigned endRow) {
                for (unsigned i = startRow; i < endRow; ++i) {
                    for (unsigned j = 0; j < this->_numberOfColumns; j++) {
                        T sum = 0.0;
                        for (unsigned k = 0; k < this->_numberOfColumns; k++) {
                            sum += scaleThis * thisMatrixDataPtr[i * this->_numberOfColumns + k] *
                                scaleOther * inputMatrixDataPtr[k * this->_numberOfColumns + j];
                        }
                    }
                }
            };

            this->_threading.executeParallelJob(matrixMultiplyJob, this->_numberOfRows);
        }

        void matrixMultiplyIntoThis(vector<NumericalMatrixStorage<T>>&inputMatrixData, T scaleThis, T scaleOther) override {

            T *&thisMatrixDataPtr = this->_values->getDataPointer();
            T *&inputMatrixDataPtr = inputMatrixData[0]._values->getDataPointer();

            auto matrixMultiplyJob = [&](unsigned startRow, unsigned endRow) {
                for (unsigned i = startRow; i < endRow; ++i) {
                    for (unsigned j = 0; j < this->_numberOfColumns; j++) {
                        T sum = 0.0;
                        for (unsigned k = 0; k < this->_numberOfColumns; k++) {
                            sum += scaleThis * thisMatrixDataPtr[i * this->_numberOfColumns + k] *
                                scaleOther * inputMatrixDataPtr[k * this->_numberOfColumns + j];
                        }
                    }
                }
            };
            
            this->_threading.executeParallelJob(matrixMultiplyJob, this->_numberOfRows);
        }
                
        void matrixMultiplyWithVector(T *&inputVectorData, T *&resultVectorData, T scaleThis, T scaleVector) override {
                
                T *&thisMatrixDataPtr = this->_values->getDataPointer();
                
                auto matrixVectorMultiplicationJob = [&](unsigned startRow, unsigned endRow) {
                    for (unsigned i = startRow; i < endRow; ++i) {
                        T sum = 0.0;
                        for (unsigned j = 0; j < this->_numberOfColumns; j++) {
                            sum += scaleThis * thisMatrixDataPtr[i * this->_numberOfColumns + j] *
                                scaleVector * inputVectorData[j];
                        }
                        resultVectorData[i] = sum;
                    }
                };
                
                this->_threading.executeParallelJob(matrixVectorMultiplicationJob, this->_numberOfRows);
        }
        
        void axpy(T *&multipliedVectorData, T *&addedVectorData, T scaleMultiplication, T scaleAddition) override {
                
                T *&thisMatrixDataPtr = this->_values->getDataPointer();
                
                auto axpyJob = [&](unsigned startRow, unsigned endRow) {
                    for (unsigned i = startRow; i < endRow; ++i) {
                        T sum = 0.0;
                        for (unsigned j = 0; j < this->_numberOfColumns; j++) {
                            sum += scaleMultiplication * thisMatrixDataPtr[i * this->_numberOfColumns + j] *
                                multipliedVectorData[j];
                        }
                        addedVectorData[i] += scaleAddition * sum;
                    }
                };
                
                this->_threading.executeParallelJob(axpyJob, this->_numberOfRows);
        }
        
    };

} // LinearAlgebra

#endif //UNTITLED_ROWMAJORMATRIXSTORAGE_H
