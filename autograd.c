#include <stdio.h>
#include "autograd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

Graph *mt_create_graph(size_t n)
{
    Graph *graph = malloc(2 * sizeof(size_t) + n * sizeof(Variable));
    graph->capcity = n;
    graph->size = 0;
    return graph;
}

Variable *mt_create_var(Graph *graph, double val)
{
    Variable *node = &graph->data[graph->size];
    *node = (Variable){
        .val = val,
        .deps_size = 0,
        .deps = {0},
        .backward = NULL,
        .grad = 0.,
        .visited = false};
    graph->size++;
    return node;
}

Variable *mt_create_var_debug(Graph *graph, double val, char *mark)
{
    Variable *node = &graph->data[graph->size];
    *node = (Variable){
        .val = val,
        .deps_size = 0,
        .deps = {0},
        .backward = NULL,
        .grad = 0.,
        .visited = false};
    strlcpy(node->debug, mark, sizeof(node->debug));
    graph->size++;
    return node;
}

void clear_grad(Variable *x)
{
    x->grad = 0.;
}

void clear_graph_grads(Graph *graph)
{
    for (size_t i = 0; i < graph->size; ++i)
    {
        clear_grad(&graph->data[i]);
    }
}

void clear_visited(Variable *x)
{
    x->visited = false;
}

void clear_graph_visiteds(Graph *graph)
{
    for (size_t i = 0; i < graph->size; ++i)
    {
        clear_visited(&graph->data[i]);
    }
}

void clear_visiteds(size_t ls_len, Variable **node_ls)
{
    for (size_t i = 0; i < ls_len; ++i)
    {
        clear_visited(node_ls[i]);
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
    Variable *node_ls[10000];
    size_t cnt = 0;
    size_t *cnt_ptr = &cnt;
    _build_dep_ordered_list_recur(res, node_ls, cnt_ptr);

    res->grad = 1.;
    _backward(node_ls, cnt);
    clear_visiteds(cnt, node_ls);
}

void mt_backward_graph(Graph *graph)
{
    graph->data[graph->size - 1].grad = 1.;
    for (int i = graph->size - 1; i >= 0; i--)
    {
        Variable *node = &graph->data[i];
        if (node->deps_size == 0)
            continue;
        node->backward(node);
    }
}

static void _backward_grad(Variable *dep, double grad)
{
    dep->grad += grad;
}

/*
 * Add
 */
void mt_add_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    _backward_grad(res->deps[0], res->grad);
    _backward_grad(res->deps[1], res->grad);
}

Variable *mt_add(Graph *graph, Variable *left, Variable *right)
{
    // computation
    double res_val = left->val + right->val;

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_add_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Sub
 */
void mt_sub_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    _backward_grad(res->deps[0], res->grad);
    _backward_grad(res->deps[1], -res->grad);
}

Variable *mt_sub(Graph *graph, Variable *left, Variable *right)
{
    // computation
    double res_val = left->val - right->val;

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_sub_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Mul
 */
void mt_mul_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    _backward_grad(res->deps[0], res->grad * res->deps[1]->val);
    _backward_grad(res->deps[1], res->grad * res->deps[0]->val);
}

Variable *mt_mul(Graph *graph, Variable *left, Variable *right)
{
    // computation
    double res_val = left->val * right->val;

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_mul_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Div
 */
void mt_div_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *left = res->deps[0];
    Variable *right = res->deps[1];

    _backward_grad(left, res->grad / right->val);
    _backward_grad(right, -res->grad * left->val / (right->val * right->val));
}

Variable *mt_div(Graph *graph, Variable *left, Variable *right)
{
    // computation
    double res_val = left->val / right->val;

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_div_backward,
        .deps_size = 2,
        .deps = {left, right},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Pow
 */
void mt_pow_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];
    double expo = res->opr_param;

    _backward_grad(dep, res->grad * expo * pow(dep->val, expo - 1));
}

Variable *mt_pow(Graph *graph, Variable *x, double expo)
{
    // computation
    double res_val = pow(x->val, expo);

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_pow_backward,
        .opr_param = expo,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Exp
 */
void mt_exp_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, res->grad * exp(dep->val));
}

Variable *mt_exp(Graph *graph, Variable *x)
{
    // computation
    double res_val = exp(x->val);

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_exp_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Log
 */
void mt_log_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, res->grad / dep->val);
}

Variable *mt_log(Graph *graph, Variable *x)
{
    // computation
    double res_val = log(x->val);

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_log_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Sin
 */
void mt_sin_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, res->grad * (cos(dep->val)));
}

Variable *mt_sin(Graph *graph, Variable *x)
{
    // computation
    double res_val = sin(x->val);

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_sin_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    graph->size++;
    return res;
}

/*
 * Cos
 */
void mt_cos_backward(Variable *res)
{
    if (res->deps_size == 0)
        return;

    Variable *dep = res->deps[0];

    _backward_grad(dep, -res->grad * (sin(dep->val)));
}

Variable *mt_cos(Graph *graph, Variable *x)
{
    // computation
    double res_val = cos(x->val);

    // meta data
    Variable *res = &graph->data[graph->size];
    *res = (Variable){
        .val = res_val,
        .backward = mt_cos_backward,
        .deps_size = 1,
        .deps = {x},
        .grad = 0.,
    };
    graph->size++;
    return res;
}