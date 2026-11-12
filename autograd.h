#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <stddef.h>
#include <stdbool.h>

typedef struct Variable Variable;
typedef struct VarArray VarArray;
typedef struct Graph Graph;

typedef void (*BackwardFunc)(Variable *);

struct Variable
{
    double val;
    double grad;
    BackwardFunc backward; // if leaf node, set as nullptr
    double opr_param;
    size_t deps_size;
    Variable *deps[2];
    bool visited;
    char debug[10];
};

struct Graph
{
    size_t capcity;
    size_t size;
    Variable data[];
};

Graph *mt_create_graph(size_t);

Variable *mt_create_var(Graph *, double);
Variable *mt_create_var_debug(Graph *, double, char *);

void clear_grad(Variable *);
void clear_graph_grads(Graph *);
void clear_grads(size_t, Variable **);
void mt_backward(Variable *);
void mt_backward_graph(Graph *);

typedef Variable (*BinaryOpr)(Variable *, Variable *);
typedef Variable (*UnaryOpr)(Variable *);
typedef Variable (*UnaryOprWithParam)(Variable *, double);

Variable *mt_add(Graph *, Variable *, Variable *);
Variable *mt_sub(Graph *, Variable *, Variable *);
Variable *mt_mul(Graph *, Variable *, Variable *);
Variable *mt_div(Graph *, Variable *, Variable *);
Variable *mt_pow(Graph *, Variable *, double);
Variable *mt_exp(Graph *, Variable *);
Variable *mt_log(Graph *, Variable *);
Variable *mt_sin(Graph *, Variable *);
Variable *mt_cos(Graph *, Variable *);

void mt_add_backward(Variable *);
void mt_sub_backward(Variable *);
void mt_mul_backward(Variable *);
void mt_div_backward(Variable *);
void mt_pow_backward(Variable *);
void mt_exp_backward(Variable *);
void mt_log_backward(Variable *);
void mt_sin_backward(Variable *);
void mt_cos_backward(Variable *);

#endif
