A = [2 1 -1  ; -4 2 3 ; -4 3 5];
b = [1 ;1 ;1];
[L,U,P] = lu(A);
x = A \ b