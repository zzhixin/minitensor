#include <stdio.h>
#include "autograd.h"
#include <math.h>

Variable mt_create_var(double val)
{
    Variable var = {
        .val = val,
        .deps_size = 0,
        .deps = {0},
        .backward = NULL,
        .grad = 0.};
    return var;
}

void clear_grad(Variable *x)
{
    x->grad = 0.;
}

void clear_grads(size_t params_size, Variable *params[2])
{
    for (size_t i = 0; i < params_size; ++i)
    {
        params[i]->grad = 0;
    }
}

// recurse backward
static void _backward(Variable *res)
{
    if (res->deps_size == 0)
        return;
    res->backward(res);
    for (size_t i = 0; i < res->deps_size; ++i)
    {
        _backward(res->deps[i]);
    }
}

void mt_backward(Variable *res)
{
    res->grad = 1.;
    _backward(res);
}

static void _backward_grad(Variable *dep, double grad)
{
    if (dep->deps_size == 0)
    {
        dep->grad += grad;
    }
    else
    {
        dep->grad = grad;
    }
}

/*
 * Add
 */
static void _add_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    _backward_grad(res->deps[0], res->grad);
    _backward_grad(res->deps[1], res->grad);
}

Variable mt_add(Variable *left, Variable *right)
{
    // computation
    double res_val = left->val + right->val;

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _add_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    return res;
}

/*
 * Sub
 */
static void _sub_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    _backward_grad(res->deps[0], res->grad);
    _backward_grad(res->deps[1], -res->grad);
}

Variable mt_sub(Variable *left, Variable *right)
{
    // computation
    double res_val = left->val - right->val;

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _sub_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    return res;
}

/*
 * Mul
 */
static void _mul_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    _backward_grad(res->deps[0], res->grad * res->deps[1]->val);
    _backward_grad(res->deps[1], res->grad * res->deps[0]->val);
}

Variable mt_mul(Variable *left, Variable *right)
{
    // computation
    double res_val = left->val * right->val;

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _mul_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    return res;
}

/*
 * Div
 */
static void _div_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *left = res->deps[0];
    Variable *right = res->deps[1];

    _backward_grad(left, res->grad / right->val);
    _backward_grad(right, -res->grad * left->val / (right->val * right->val));
}

Variable mt_div(Variable *left, Variable *right)
{
    // computation
    double res_val = left->val / right->val;

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _div_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    return res;
}

/*
 * Pow
 */
static void _pow_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];
    double expo = res->opr_param;

    _backward_grad(dep, res->grad * expo * pow(dep->val, expo - 1));
}

Variable mt_pow(Variable *x, double expo)
{
    // computation
    double res_val = pow(x->val, expo);

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _pow_backward,
        .opr_param = expo,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    return res;
}

/*
 * Exp
 */
static void _exp_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, res->grad * exp(dep->val));
}

Variable mt_exp(Variable *x)
{
    // computation
    double res_val = exp(x->val);

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _exp_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    return res;
}

/*
 * Log
 */
static void _log_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, res->grad / dep->val);
}

Variable mt_log(Variable *x)
{
    // computation
    double res_val = log(x->val);

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _log_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    return res;
}

/*
 * Sin
 */
static void _sin_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, res->grad * (cos(dep->val)));
}

Variable mt_sin(Variable *x)
{
    // computation
    double res_val = sin(x->val);

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _sin_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    return res;
}

/*
 * Cos
 */
static void _cos_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, -res->grad * (sin(dep->val)));
}

Variable mt_cos(Variable *x)
{
    // computation
    double res_val = cos(x->val);

    // meta data
    Variable res = {
        .val = res_val,
        .backward = _cos_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    return res;
}