//
// Created by hal9000 on 9/2/23.
//

#ifndef UNTITLED_NUMERICALMATRIXSTORAGE_H
#define UNTITLED_NUMERICALMATRIXSTORAGE_H
#include "../../NumericalVector/NumericalVector.h"
#include <list>
namespace LinearAlgebra {
    enum NumericalMatrixStorageType {
        FullMatrix,
        CoordinateList,
        CSR,
        
    };
    template <typename T>
    class NumericalMatrixStorage {
    public:
        explicit NumericalMatrixStorage(NumericalMatrixStorageType storageType, unsigned numberOfRows, unsigned numberOfColumns, 
                                        ParallelizationMethod parallelizationMethod, unsigned availableThreads = 1) :
        _storageType(storageType), _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns),
        threading(ThreadingOperations<T>(parallelizationMethod, availableThreads)) { }
        
        virtual ~NumericalMatrixStorage() = default;
        
        shared_ptr<NumericalVector<T> >& getValues(){
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            return _values;
        }
        
        virtual vector<shared_ptr<NumericalVector<T>>>& getSupplementaryVectors(){ }
        
        const NumericalMatrixStorageType& getStorageType(){
            return _storageType;
        }

        ThreadingOperations<T> threading;


        virtual T&  getElement(unsigned row, unsigned column) {}

        virtual void setElement(unsigned row, unsigned column, const T &value) {}

        virtual void eraseElement(unsigned row, unsigned column, const T &value) {}

        virtual shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) {}
        
        virtual shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) {}

        virtual void getRow(unsigned row, shared_ptr<NumericalVector<T>> &vector) {}
        
        virtual void getColumn(unsigned column, shared_ptr<NumericalVector<T>> &vector) {}

        virtual void initializeElementAssignment() {}

        virtual void finalizeElementAssignment() {}
        
        virtual void matrixAdd(NumericalMatrixStorage<T> &inputMatrixData,
                               NumericalMatrixStorage<T> &resultMatrixData,
                               T scaleThis, T scaleOther) {}
        
        virtual void matrixAddIntoThis(NumericalMatrixStorage<T> &inputMatrixData,
                                       T scaleThis, T scaleOther) {}
        
        virtual void matrixSubtract(NumericalMatrixStorage<T> &inputMatrixData,
                                    NumericalMatrixStorage<T> &resultMatrixData,
                                    T scaleThis, T scaleOther) {}
        
        virtual void matrixSubtractFromThis(NumericalMatrixStorage<T> &inputMatrixData,
                                            T scaleThis, T scaleOther) {}
        
        virtual void matrixMultiply(NumericalMatrixStorage<T> &inputMatrixData,
                                    NumericalMatrixStorage<T> &resultMatrixData,
                                    T scaleThis, T scaleOther) {}
        
        virtual void matrixMultiplyIntoThis(NumericalMatrixStorage<T> &inputMatrixData,
                                            T scaleThis, T scaleOther) {}
        
        virtual void matrixMultiplyWithVector(T *&inputVectorData, T *&resultVectorData, T scaleThis, T scaleOther) {}
        
        virtual void axpy(T *&multipliedVectorData, T *&addedVectorData, T scaleMultiplication, T scaleAddition) {}


    protected:
        
        shared_ptr<NumericalVector<T> > _values;
        
        NumericalMatrixStorageType _storageType;
        
        unsigned _numberOfRows;
        
        unsigned _numberOfColumns;
        
        bool _elementAssignmentRunning;
        
        virtual unsigned _numberOfNonZeroElements() {
            return -1;
        }

        virtual unsigned _numberOfZeroElements() {
            return -1;
        }

        virtual double _density() {
            return -1;
        }
    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIXSTORAGE_H
