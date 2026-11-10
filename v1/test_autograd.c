#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "autograd.h"

void test_add()
{
    Variable x1 = mt_create_var(1.);
    Variable x2 = mt_create_var(3.);
    Variable y = mt_add(&x1, &x2);
    clear_grad(&x1);
    clear_grad(&x2);
    mt_backward(&y);
    assert(y.val == x1.val + x2.val);
    assert(x1.grad == 1. && x2.grad == 1.);
}

void test_sub()
{
    Variable x1 = mt_create_var(1.);
    Variable x2 = mt_create_var(3.);
    Variable y = mt_sub(&x1, &x2);
    clear_grad(&x1);
    clear_grad(&x2);
    mt_backward(&y);
    assert(y.val == x1.val - x2.val);
    assert(x1.grad == 1. && x2.grad == -1.);
}

void test_mul()
{
    Variable x1 = mt_create_var(1.);
    Variable x2 = mt_create_var(3.);
    Variable y = mt_mul(&x1, &x2);
    clear_grad(&x1);
    clear_grad(&x2);
    mt_backward(&y);
    assert(y.val == x1.val * x2.val);
    assert(x1.grad == 3. && x2.grad == 1.);
}

void test_div()
{
    Variable x1 = mt_create_var(3.);
    Variable x2 = mt_create_var(2.);
    Variable y = mt_div(&x1, &x2);
    clear_grad(&x1);
    clear_grad(&x2);
    mt_backward(&y);
    assert(y.val == x1.val / x2.val);
    assert(x1.grad == 1. / 2. && x2.grad == -3. / (2. * 2.));
}

void test_pow()
{
    Variable x = mt_create_var(3.);
    Variable y = mt_pow(&x, 5);
    clear_grad(&x);
    mt_backward(&y);
    assert(y.val == pow(3., 5.));
    assert(x.grad == 5. * pow(3., 4.));
}

void test_exp()
{
    Variable x1 = mt_create_var(3.);
    Variable y = mt_exp(&x1);
    clear_grad(&x1);
    mt_backward(&y);
    assert(y.val == exp(3.));
    assert(x1.grad == y.val);
}

void test_log()
{
    Variable x = mt_create_var(3.);
    Variable y = mt_log(&x);
    clear_grad(&x);
    mt_backward(&y);
    assert(y.val == log(3.));
    assert(x.grad == 1 / x.val);
}

void test_sin()
{
    Variable x = mt_create_var(3.);
    Variable y = mt_sin(&x);
    clear_grad(&x);
    mt_backward(&y);
    assert(y.val == sin(3.));
    assert(x.grad == cos(3.));
}

void test_cos()
{
    Variable x = mt_create_var(3.);
    Variable y = mt_cos(&x);
    clear_grad(&x);
    mt_backward(&y);
    assert(y.val == cos(3.));
    assert(x.grad == -sin(3.));
}

void test_exp_log()
{
    Variable x = mt_create_var(3.);
    Variable y = mt_exp(&x);
    Variable z = mt_log(&y);
    clear_grad(&x);
    mt_backward(&z);
    assert(z.val == x.val);
    assert(x.grad == 1.);
}

void test_compound_opr()
{
    Variable a = mt_create_var(2.);
    Variable b = mt_create_var(3.);
    Variable c = mt_sin(&a);
    Variable d = mt_div(&c, &b);
    Variable e = mt_exp(&d);
    Variable f = mt_add(&c, &e);
    clear_grads(6, (Variable *[]){&a, &b, &d, &e, &c, &f});
    mt_backward(&f);
    assert(f.val == sin(a.val) + exp(sin(a.val) / b.val));
    assert(a.grad == cos(a.val) + exp(sin(a.val) / b.val) * cos(a.val) / b.val);
    assert(b.grad == -exp(sin(a.val) / b.val) * sin(a.val) / pow(b.val, 2));
}

int main()
{
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
    test_compound_opr();
    printf("All tests passed!\n");
}
