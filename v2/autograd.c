#include <stdio.h>
#include "autograd.h"
#include <math.h>

Variable mt_create_var(float val)
{
    Variable var = {
        .val = val,
        .deps_size = 0,
        .deps = {0},
        .backward = NULL,
        .grad = 0.,
        .visited = false};
    return var;
}

Variable mt_create_var_debug(float val, char name)
{
    Variable var = {
        .val = val,
        .deps_size = 0,
        .deps = {0},
        .backward = NULL,
        .grad = 0.,
        .visited = false,
        .debug = name};
    return var;
}

void clear_grad(Variable *x)
{
    x->grad = 0.;
}

void clear_grads(size_t vars_size, Variable **vars)
{
    for (size_t i = 0; i < vars_size; ++i)
    {
        clear_grad(vars[i]);
    }
}

void clear_visited(Variable *x)
{
    x->visited = false;
}

void clear_visiteds(size_t vars_size, Variable **vars)
{
    for (size_t i = 0; i < vars_size; ++i)
    {
        clear_visited(vars[i]);
    }
}

static void _build_dep_ordered_list_recur(Variable *node, Variable **node_ls, size_t *cnt)
{
    if (node->visited)
        return;
    if (node->deps_size > 0)
    {
        for (size_t i = 0; i < node->deps_size; ++i)
        {
            _build_dep_ordered_list_recur(node->deps[i], node_ls, cnt);
        }
    }
    node_ls[*cnt] = node;
    (*cnt)++;
    node->visited = true;
}

static void _backward(Variable **node_ls, size_t len)
{
    for (int i = len - 1; i >= 0; i--)
    {
        Variable *node = node_ls[i];
        if (node->deps_size == 0)
            continue;
        node->backward(node);
    }
}

void mt_backward(Variable *res)
{
    // build dependent order list by last DFS
    Variable *node_ls[100];
    size_t cnt = 0;
    size_t *cnt_ptr = &cnt;
    _build_dep_ordered_list_recur(res, node_ls, cnt_ptr);

    res->grad = 1.;
    _backward(node_ls, cnt);
    clear_visiteds(cnt, node_ls);
}

static void _backward_grad(Variable *dep, float grad)
{
    dep->grad += grad;
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
    float res_val = left->val + right->val;

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
    float res_val = left->val - right->val;

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
    float res_val = left->val * right->val;

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
    float res_val = left->val / right->val;

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
    float expo = res->opr_param;

    _backward_grad(dep, res->grad * expo * powf(dep->val, expo - 1));
}

Variable mt_pow(Variable *x, float expo)
{
    // computation
    float res_val = powf(x->val, expo);

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

    _backward_grad(dep, res->grad * expf(dep->val));
}

Variable mt_exp(Variable *x)
{
    // computation
    float res_val = expf(x->val);

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
    float res_val = logf(x->val);

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

    _backward_grad(dep, res->grad * (cosf(dep->val)));
}

Variable mt_sin(Variable *x)
{
    // computation
    float res_val = sinf(x->val);

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

    _backward_grad(dep, -res->grad * (sinf(dep->val)));
}

Variable mt_cos(Variable *x)
{
    // computation
    float res_val = cosf(x->val);

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