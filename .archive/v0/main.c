#include <stdio.h>
#include "autograd.h"


int main() {
    Variable x1 = {
        .val = 1.,
        .deps = {0},
        .opr = NULL,
        .grad = 0.
    };

    Variable x2 = {
        .val = 3.,
        .deps = {0},
        .opr = NULL,
        .grad = 0.
    };
    printf("x1: %.2f, x2: %.2f\n\n", x1.val, x2.val);

    Variable y = Prod.forward((VarArray){2, (Variable*[]){&x1, &x2}});
    printf("y = x1 * x2: %.2f\n", y.val);
    clear_grad(&x1); clear_grad(&x2);
    backward(&y);
    printf("x1.grad: %.2f, x2.grad: %.2f\n\n", x1.grad, x2.grad);

    Variable z = Sum.forward((VarArray){2, (Variable*[]){&y, &x2}});
    printf("z = x1 * x2 + x2: %.2f\n", z.val);
    clear_grad(&x1); clear_grad(&x2);
    backward(&z);
    printf("x1.grad: %.2f, x2.grad: %.2f\n\n", x1.grad, x2.grad);


    Variable w = Div.forward((VarArray){2, (Variable*[]){&x1, &x2}});
    printf("w = x1 / x2: %.2f\n", w.val);
    clear_grad(&x1); clear_grad(&x2);
    backward(&w);
    printf("x1.grad: %.2f, x2.grad: %.2f\n", x1.grad, x2.grad);
}
