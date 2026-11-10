#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <stddef.h>

typedef struct Operator Operator;
typedef struct Variable Variable;
typedef struct VarArray VarArray; 

typedef Variable (*ForwardFunc) (VarArray);
typedef void (*BackwardFunc) (Variable*);

// VarArray is a temporary wrapper, it doesn't not hold anything.
// Even when you create a VarArray from some Variable, double pointer 
// makes the data point to the original Variable. 
// If implemented as `Variable* data, the pointer actually points to the 
// array you create later. 
struct VarArray
{
    size_t len;
    Variable** data;   // VarArray point to the data in the original seperate var. 
};

struct Variable
{
    float val;
    VarArray deps;
    Operator* opr;  // if leaf node, set as nullptr
    float grad;
};

struct Operator
{
    ForwardFunc forward;
    BackwardFunc backward;
};

void clear_grad(Variable*); 

void clear_grads(VarArray); 

void backward(Variable*);

extern Operator Sum;
extern Operator Sub;
extern Operator Prod;
extern Operator Div;


#endif
