#include <stdio.h>
#include "autograd.h"


int main() {
    Variable x1 = mt_create_var(1.);
    Variable x2 = mt_create_var(3.);
    printf("x1: %.2f, x2: %.2f\n\n", x1.val, x2.val);

    Variable temp = mt_mul(&x1, &x2);
    Variable z = mt_add(&temp, &x2);
    printf("z = x1 * x2 + x2: %.2f\n", z.val);
    clear_grad(&x1); clear_grad(&x2);
    mt_backward(&z);
    printf("x1.grad: %.2f, x2.grad: %.2f\n\n", x1.grad, x2.grad);
}