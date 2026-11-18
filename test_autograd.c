#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "autograd.h"
#include <string.h>

void test_add()
{

    Graph *graph = mt_create_graph(100);
    Variable *x1 = mt_create_var(graph, 1.);
    Variable *x2 = mt_create_var(graph, 3.);
    Variable *y = mt_add(graph, x1, x2);

    clear_grad(x1);
    clear_grad(x2);
    mt_backward(y);
    assert(y->val == x1->val + x2->val);
    assert(x1->grad == 1. && x2->grad == 1.);

    clear_grad(x1);
    clear_grad(x2);
    mt_backward_graph(graph);
    assert(y->val == x1->val + x2->val);
    assert(x1->grad == 1. && x2->grad == 1.);
}

void test_sub()
{

    Graph *graph = mt_create_graph(100);
    Variable *x1 = mt_create_var(graph, 1.);
    Variable *x2 = mt_create_var(graph, 3.);
    Variable *y = mt_sub(graph, x1, x2);

    clear_grad(x1);
    clear_grad(x2);
    mt_backward(y);
    assert(y->val == x1->val - x2->val);
    assert(x1->grad == 1. && x2->grad == -1.);

    clear_grad(x1);
    clear_grad(x2);
    mt_backward_graph(graph);
    assert(y->val == x1->val - x2->val);
    assert(x1->grad == 1. && x2->grad == -1.);
}

void test_mul()
{

    Graph *graph = mt_create_graph(100);
    Variable *x1 = mt_create_var(graph, 1.);
    Variable *x2 = mt_create_var(graph, 3.);
    Variable *y = mt_mul(graph, x1, x2);

    clear_grad(x1);
    clear_grad(x2);
    mt_backward(y);
    assert(y->val == x1->val * x2->val);
    assert(x1->grad == x2->val && x2->grad == x1->val);

    clear_grad(x1);
    clear_grad(x2);
    mt_backward_graph(graph);
    assert(y->val == x1->val * x2->val);
    assert(x1->grad == x2->val && x2->grad == x1->val);
}

void test_div()
{

    Graph *graph = mt_create_graph(100);
    Variable *x1 = mt_create_var(graph, 3.);
    Variable *x2 = mt_create_var(graph, 4.);
    Variable *y = mt_div(graph, x1, x2);

    clear_grad(x1);
    clear_grad(x2);
    mt_backward(y);
    assert(y->val == x1->val / x2->val);
    assert(x1->grad == 1. / x2->val && x2->grad == -x1->val / (x2->val * x2->val));

    clear_grad(x1);
    clear_grad(x2);
    mt_backward_graph(graph);
    assert(y->val == x1->val / x2->val);
    assert(x1->grad == 1. / x2->val && x2->grad == -x1->val / (x2->val * x2->val));
}

void test_pow()
{

    Graph *graph = mt_create_graph(100);
    Variable *x = mt_create_var(graph, 3.);
    Variable *y = mt_pow(graph, x, 5);

    clear_grad(x);
    mt_backward(y);
    assert(y->val == pow(x->val, 5));
    assert(x->grad == 5. * pow(x->val, 4.));

    clear_grad(x);
    mt_backward_graph(graph);
    assert(y->val == pow(x->val, 5));
    assert(x->grad == 5. * pow(x->val, 4.));
}

void test_exp()
{
    Graph *graph = mt_create_graph(100);
    Variable *x = mt_create_var(graph, 3.);
    Variable *y = mt_exp(graph, x);

    clear_grad(x);
    mt_backward(y);
    assert(y->val == exp(x->val));
    assert(x->grad == y->val);

    clear_grad(x);
    mt_backward_graph(graph);
    assert(y->val == exp(x->val));
    assert(x->grad == y->val);
}

void test_log()
{
    Graph *graph = mt_create_graph(100);
    Variable *x = mt_create_var(graph, 3.);
    Variable *y = mt_log(graph, x);

    clear_grad(x);
    mt_backward(y);
    assert(y->val == log(x->val));
    assert(x->grad == 1 / x->val);

    clear_grad(x);
    mt_backward_graph(graph);
    assert(y->val == log(x->val));
    assert(x->grad == 1 / x->val);
}

void test_sin()
{
    Graph *graph = mt_create_graph(100);
    Variable *x = mt_create_var(graph, 3.);
    Variable *y = mt_sin(graph, x);

    clear_grad(x);
    mt_backward(y);
    assert(y->val == sin(x->val));
    assert(x->grad == cos(x->val));

    clear_grad(x);
    mt_backward_graph(graph);
    assert(y->val == sin(x->val));
    assert(x->grad == cos(x->val));
}

void test_cos()
{
    Graph *graph = mt_create_graph(100);
    Variable *x = mt_create_var(graph, 3.);
    Variable *y = mt_cos(graph, x);

    clear_grad(x);
    mt_backward(y);
    assert(y->val == cos(x->val));
    assert(x->grad == -sin(x->val));

    clear_grad(x);
    mt_backward_graph(graph);
    assert(y->val == cos(x->val));
    assert(x->grad == -sin(x->val));
}

void test_exp_log()
{
    Graph *graph = mt_create_graph(100);
    Variable *x = mt_create_var(graph, 3.);
    Variable *y = mt_exp(graph, x);
    Variable *z = mt_log(graph, y);

    clear_graph_grads(graph);
    mt_backward(z);
    assert(z->val == x->val);
    assert(x->grad == 1.);

    clear_graph_grads(graph);
    mt_backward_graph(graph);
    assert(z->val == x->val);
    assert(x->grad == 1.);
}

void test_compound_opr_complex()
{
    Graph *graph = mt_create_graph(100);
    Variable *a = mt_create_var_debug(graph, 2., "a");
    Variable *b = mt_create_var_debug(graph, 3., "b");
    Variable *c = mt_sin(graph, a);
    strlcpy(c->debug, "c", sizeof(c->debug));
    Variable *d = mt_div(graph, c, b);
    strlcpy(d->debug, "d", sizeof(c->debug));
    Variable *e = mt_exp(graph, d);
    strlcpy(e->debug, "e", sizeof(c->debug));
    Variable *f = mt_add(graph, c, e);
    strlcpy(f->debug, "f", sizeof(c->debug));

    clear_graph_grads(graph);
    mt_backward(f);
    assert(f->val == sin(a->val) + exp(sin(a->val) / b->val));
    assert(e->grad == f->grad);
    assert(d->grad == e->val);
    assert(c->grad == f->grad + d->grad / b->val);
    assert(b->grad == -exp(sin(a->val) / b->val) * sin(a->val) / pow(b->val, 2));
    assert(a->grad == cos(a->val) + exp(sin(a->val) / b->val) * cos(a->val) / b->val);

    clear_graph_grads(graph);
    mt_backward_graph(graph);
    assert(f->val == sin(a->val) + exp(sin(a->val) / b->val));
    assert(e->grad == f->grad);
    assert(d->grad == e->val);
    assert(c->grad == f->grad + d->grad / b->val);
    assert(b->grad == -exp(sin(a->val) / b->val) * sin(a->val) / pow(b->val, 2));
    assert(a->grad == cos(a->val) + exp(sin(a->val) / b->val) * cos(a->val) / b->val);
}

int main()
{
    printf("> test_autograd: start test\n");
    test_add();
    test_sub();
    test_mul();
    test_div();
    test_pow();
    test_exp();
    test_log();
    test_cos();
    test_sin();
    test_exp_log();
    test_compound_opr_complex();
    printf("  test_autograd: all tests passed!\n");
}
