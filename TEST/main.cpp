#include <iostream>
#include "Array.h"

int main() {
    auto A = new Array<double>(3, 3);
    A->populateElement(0, 1, new double(2.0));

   
    A->print();
    
    return 0;
}
