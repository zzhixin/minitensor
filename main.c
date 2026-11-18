#include <stdio.h>
#include "autograd.h"
#include "tensor.h"

void run_autograd()
{
    Graph *graph = mt_create_graph(100);

    Variable *x1 = mt_create_var(graph, 1.);
    Variable *x2 = mt_create_var(graph, 3.);
    printf("x1: %.2f, x2: %.2f\n\n", x1->val, x2->val);

    Variable *z = mt_add(graph, mt_mul(graph, x1, x2), x2);
    printf("z = x1 * x2 + x2: %.2f\n", z->val);
    clear_grad(x1);
    clear_grad(x2);
    mt_backward(z);
    printf("x1.grad: %.2f, x2.grad: %.2f\n\n", x1->grad, x2->grad);

    printf("x1.visited: %d\n", x1->visited);
}

void run_tensor()
{
    Vec *v = mt_create_vec_by(4, (double[]){0., 1., 2., 3.});
    mt_print_vec(v);
}

int main()
{
    // run_autograd();
    run_tensor();
}