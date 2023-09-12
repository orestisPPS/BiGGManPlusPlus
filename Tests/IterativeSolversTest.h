//
// Created by hal9000 on 9/12/23.
//

#ifndef UNTITLED_ITERATIVESOLVERSTEST_H
#define UNTITLED_ITERATIVESOLVERSTEST_H


#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/SORSolver.h"
#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/GaussSeidelSolver.h"
#include "../LinearAlgebra/Solvers/Iterative/StationaryIterative/JacobiSolver.h"
#include "../LinearAlgebra/Solvers/Iterative/GradientBasedIterative/ConjugateGradientSolver.h"

namespace Tests{
    
    class IterativeSolversTest {
    public:
        static void runTests(){
            
            auto matrix = make_shared<NumericalMatrix<double>>(5,5);
            auto vector = make_shared<NumericalVector<double>>(5);
            auto solution = make_shared<NumericalVector<double>>(5);

            // Setting matrix values
            matrix->setElement(0,0, 1.5992);
            matrix->setElement(0,1, 1.0903);
            matrix->setElement(0,2, 0.8860);
            matrix->setElement(0,3, 1.5552);
            matrix->setElement(0,4, 1.1086);

            matrix->setElement(1,0, 1.0903);
            matrix->setElement(1,1, 1.5611);
            matrix->setElement(1,2, 0.5546);
            matrix->setElement(1,3, 1.1585);
            matrix->setElement(1,4, 0.7540);

            matrix->setElement(2,0, 0.8860);
            matrix->setElement(2,1, 0.5546);
            matrix->setElement(2,2, 0.5430);
            matrix->setElement(2,3, 0.8553);
            matrix->setElement(2,4, 0.6748);

            matrix->setElement(3,0, 1.5552);
            matrix->setElement(3,1, 1.1585);
            matrix->setElement(3,2, 0.8553);
            matrix->setElement(3,3, 1.6416);
            matrix->setElement(3,4, 1.3566);

            matrix->setElement(4,0, 1.1086);
            matrix->setElement(4,1, 0.7540);
            matrix->setElement(4,2, 0.6748);
            matrix->setElement(4,3, 1.3566);
            matrix->setElement(4,4, 1.7652);

            // Setting vector values
            (*vector)[0] = 2.6193;
            (*vector)[1] = 2.1695;
            (*vector)[2] = 1.4703;
            (*vector)[3] = 2.8009;
            (*vector)[4] = 2.4817;

            // Setting solution values
            (*solution)[0] = 0.2297;
            (*solution)[1] = 0.4083;
            (*solution)[2] = 0.2923;
            (*solution)[3] = 0.6631;
            (*solution)[4] = 0.4659;
            
            auto linearSystem = make_shared<LinearSystem>(matrix, vector);

            _conjugateGradientTest(linearSystem, solution);
/*            _sorTest(linearSystem, solution);
            _jacobiTest(linearSystem, solution);
            _gaussSeidelTest(linearSystem, solution);*/
        }


    
    private:
        
        static void _conjugateGradientTest(shared_ptr<LinearSystem>& linearSystem, shared_ptr<NumericalVector<double>>& solution){
            auto solver = make_shared<ConjugateGradientSolver>();
            solver->setLinearSystem(linearSystem);
            solver->solve();

            auto difference = NumericalVector<double>(solution->size());
            for (auto i = 0; i < solution->size(); ++i){
                difference[i] = (*solution)[i] - (*linearSystem->solution)[i];
            }
            auto norm = difference.norm(LInf);
            cout << "Conjugate Gradient Test: " << norm << endl;
        }
        
        static void _jacobiTest(shared_ptr<LinearSystem>& linearSystem, shared_ptr<NumericalVector<double>> &solution){
            auto solver = make_shared<JacobiSolver>();
            solver->setLinearSystem(linearSystem);
            solver->solve();

            auto difference = NumericalVector<double>(solution->size());
            for (auto i = 0; i < solution->size(); ++i){
                difference[i] = (*solution)[i] - (*linearSystem->solution)[i];
            }
            auto norm = difference.norm(LInf);
            cout << "Jacobi Test: " << norm << endl;
        }
        
        static void _gaussSeidelTest(shared_ptr<LinearSystem>& linearSystem, shared_ptr<NumericalVector<double>> &solution){
            auto solver = make_shared<GaussSeidelSolver>();
            solver->setLinearSystem(linearSystem);
            solver->solve();

            auto difference = NumericalVector<double>(solution->size());
            for (auto i = 0; i < solution->size(); ++i){
                difference[i] = (*solution)[i] - (*linearSystem->solution)[i];
            }
            auto norm = difference.norm(LInf);
            cout << "Gauss-Seidel Test: " << norm << endl;
        }
        
        static void _sorTest(shared_ptr<LinearSystem>& linearSystem, shared_ptr<NumericalVector<double>> &solution){
            auto solver = make_shared<SORSolver>(1.8);
            solver->setLinearSystem(linearSystem);
            solver->solve();

            auto difference = NumericalVector<double>(solution->size());
            for (auto i = 0; i < solution->size(); ++i){
                difference[i] = (*solution)[i] - (*linearSystem->solution)[i];
            }
            auto norm = difference.norm(LInf);
            cout << "SOR Test: " << norm << endl;
        }
    };
}

#endif //UNTITLED_ITERATIVESOLVERSTEST_H
