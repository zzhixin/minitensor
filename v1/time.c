#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "autograd.h"

static const double TIMING_RUNS = 1000;

void time_simple_graph()
{
    Variable a = mt_create_var(2.);
    Variable b = mt_create_var(3.);
    Variable c = mt_sin(&a);
    Variable d = mt_div(&c, &b);
    Variable e = mt_exp(&d);
    Variable f = mt_add(&c, &e);
    Variable *vars_ls[] = {&a, &b, &d, &e, &c, &f};

    clock_t start = clock();
    for (size_t i = 0; i < TIMING_RUNS; i++)
    {
        clear_grads(6, vars_ls);
        mt_backward(&f);
    }
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC / TIMING_RUNS;
    printf("time spent: %.4f us\n", time_spent * 1e6);

    assert(f.val == sinf(a.val) + expf(sinf(a.val) / b.val));
    assert(b.grad == -expf(sinf(a.val) / b.val) * sinf(a.val) / powf(b.val, 2));
    assert(a.grad == cosf(a.val) + expf(sinf(a.val) / b.val) * cosf(a.val) / b.val);
}

int main()
{
    time_simple_graph();
}
