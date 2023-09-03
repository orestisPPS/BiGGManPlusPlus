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
                                        ParallelizationMethod parallelizationMethod) :
        _storageType(storageType), _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns), _threading(ThreadingOperations<T>(parallelizationMethod)),
        _parallelizationMethod(parallelizationMethod), _values(nullptr), _elementAssignmentRunning(false) { }
        
        virtual ~NumericalMatrixStorage() = default;
        
        const shared_ptr<NumericalVector<T> >& getValues(){
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            return _values;
        }
        
        unsigned numberOfNonZeroElements(){
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            return _values->size();
        }
        
        virtual vector<const NumericalVector<T>&> getNecessaryStorageVectors() = 0;
        
        virtual void initializeElementAssignment() = 0;
        
        virtual shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) = 0;
        
        virtual void getRow(unsigned row, shared_ptr<NumericalVector<T>> &vector) = 0;
        
        virtual shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) = 0;
        
        virtual void getColumn(unsigned column, shared_ptr<NumericalVector<T>> &vector) = 0;

        virtual void finalizeElementAssignment() = 0;


        virtual T&  getElement(unsigned row, unsigned column) = 0;
        
        virtual void setElement(unsigned row, unsigned column, const T &value) = 0;
        
        virtual void eraseElement(unsigned row, unsigned column, const T &value) = 0;
        
        virtual void scale(T scalar){
            auto scaleJob = [&](unsigned start, unsigned end) -> void {
                for (unsigned i = start; i < end; ++i) {
                    (*_values)[i] *= scalar;
                }
            };
            _threading.executeParallelJob(scaleJob, _values->size());
        }
        
        virtual void matrixAdd(NumericalMatrixStorage<T> &inputMatrixData,
                               NumericalMatrixStorage<T> &resultMatrixData,
                               T scaleThis, T scaleOther) = 0;
        
        virtual void matrixAddIntoThis(NumericalMatrixStorage<T> &inputMatrixData,
                                       T scaleThis, T scaleOther) = 0;
        
        virtual void matrixSubtract(NumericalMatrixStorage<T> &inputMatrixData,
                                    NumericalMatrixStorage<T> &resultMatrixData,
                                    T scaleThis, T scaleOther) = 0;
        
        virtual void matrixSubtractFromThis(NumericalMatrixStorage<T> &inputMatrixData,
                                            T scaleThis, T scaleOther) = 0;
        
        virtual void matrixMultiply(NumericalMatrixStorage<T> &inputMatrixData,
                                    NumericalMatrixStorage<T> &resultMatrixData,
                                    T scaleThis, T scaleOther) = 0;
        
        virtual void matrixMultiplyIntoThis(NumericalMatrixStorage<T> &inputMatrixData,
                                            T scaleThis, T scaleOther) = 0;
        
        virtual void matrixMultiplyWithVector(T *&inputVectorData, T *&resultVectorData, T scaleThis, T scaleOther) = 0;
        
        virtual void axpy(T *&multipliedVectorData, T *&addedVectorData, T scaleMultiplication, T scaleAddition) = 0;


    protected:
        
        shared_ptr<NumericalVector<T> > _values;
        
        NumericalMatrixStorageType _storageType;
        
        ParallelizationMethod _parallelizationMethod;
        
        ThreadingOperations<T> _threading;
        
        unsigned _numberOfRows;
        
        unsigned _numberOfColumns;
        
        bool _elementAssignmentRunning;
    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIXSTORAGE_H
