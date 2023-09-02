//
// Created by hal9000 on 9/2/23.
//

#ifndef UNTITLED_NUMERICALMATRIXSTORAGE_H
#define UNTITLED_NUMERICALMATRIXSTORAGE_H
#include "../../NumericalVector/NumericalVector.h"
#include <list>
namespace LinearAlgebra {
    enum NumericalMatrixStorageType {
        RowMajor,
        CoordinateList,
        CSR,
        
    };
    template <typename T>
    class NumericalMatrixStorage {
    public:
        explicit NumericalMatrixStorage(NumericalMatrixStorageType &storageType, unsigned &numberOfRows, unsigned &numberOfColumns, double sparsityPercentage = 0.0) :
        _storageType(storageType), _numberOfRows(numberOfRows), _numberOfColumns(numberOfColumns),
        _elementAssignmentRunning(false), _sparsityPercentage(sparsityPercentage) { }
        
        virtual ~NumericalMatrixStorage() = default;
        
        const NumericalMatrixStorageType& getStorageType(){
            return _storageType;
        }
        
        const shared_ptr<NumericalVector<T> >& getValues(){
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            return _values;
        }

        virtual const T& operator()(unsigned int row, unsigned int column) const = 0;
        
        unsigned numberOfNonZeroElements(){
            if (_elementAssignmentRunning){
                throw runtime_error("Element assignment is still running. Call finalizeElementAssignment() first.");
            }
            return _values->size();
        }

        virtual void initializeElementAssignment() = 0;
        
        virtual shared_ptr<NumericalVector<T>> getRowSharedPtr(unsigned row) = 0;
        
        virtual void getRow(unsigned row, shared_ptr<NumericalVector<T>> &vector) = 0;
        
        virtual shared_ptr<NumericalVector<T>> getColumnSharedPtr(unsigned column) = 0;
        
        virtual void getColumn(unsigned column, shared_ptr<NumericalVector<T>> &vector) = 0;

        virtual void finalizeElementAssignment() = 0;
        
        virtual void setElement(unsigned row, unsigned column, const T &value) = 0;


    protected:
        
        shared_ptr<NumericalVector<T> > _values;
        
        NumericalMatrixStorageType &_storageType;
        
        double _sparsityPercentage;
        
        unsigned &_numberOfRows;
        
        unsigned &_numberOfColumns;
        
        bool _elementAssignmentRunning;
        
        list<unique_ptr<list<T>>> _storageTypeSpecificLists;

    };

} // LinearAlgebra

#endif //UNTITLED_NUMERICALMATRIXSTORAGE_H
