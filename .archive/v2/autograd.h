#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <stddef.h>
#include <stdbool.h>

typedef struct Variable Variable;
typedef struct VarArray VarArray;

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

Variable mt_create_var(double);
Variable mt_create_var_debug(double, char *);

void clear_grad(Variable *);
void clear_grads(size_t, Variable *[2]);
void mt_backward(Variable *);

typedef Variable (*BinaryOpr)(Variable *, Variable *);
typedef Variable (*UnaryOpr)(Variable *);
typedef Variable (*UnaryOprWithParam)(Variable *, double);

Variable mt_add(Variable *, Variable *);
Variable mt_sub(Variable *, Variable *);
Variable mt_mul(Variable *, Variable *);
Variable mt_div(Variable *, Variable *);
Variable mt_pow(Variable *, double);
Variable mt_exp(Variable *);
Variable mt_log(Variable *);
Variable mt_sin(Variable *);
Variable mt_cos(Variable *);

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
