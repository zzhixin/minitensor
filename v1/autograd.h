#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <stddef.h>

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
};

Variable mt_create_var(double);

void clear_grad(Variable *);
void clear_grads(size_t, Variable *[2]);
void mt_backward(Variable *);

Variable mt_add(Variable *, Variable *);
Variable mt_sub(Variable *, Variable *);
Variable mt_mul(Variable *, Variable *);
Variable mt_div(Variable *, Variable *);
Variable mt_pow(Variable *, double);
Variable mt_exp(Variable *);
Variable mt_log(Variable *);
Variable mt_sin(Variable *);
Variable mt_cos(Variable *);

#endif
