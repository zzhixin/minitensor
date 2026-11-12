#include <stdio.h>
#include "autograd.h"


void clear_grad(Variable* x) {
    x->grad = 0.;
}

void clear_grads(VarArray xs) {
    for(size_t i=0; i<xs.len; ++i) {
        xs.data[i]->grad = 0;
    }
}

// recurse backward
void _backward(Variable* res) {
    if(res->opr == NULL) return;
    res->opr->backward(res);
    VarArray deps = res->deps;
    for(size_t i=0; i<deps.len; ++i) {
        _backward(deps.data[i]);
    }
}

void backward(Variable* res) {
    res->grad = 1.;
    _backward(res);
}

void _backward_grad(Variable* dep, float grad) {
    if(dep->opr == NULL) {
        dep->grad += grad;
    } else {
        dep->grad = grad;
    }
}

/*
* Sum
*/
Variable _sum_forward(VarArray params) {
    Variable** _params = params.data;
    float res_val = 0.;
    for(size_t i=0; i<params.len; i++) {
        res_val += _params[i]->val;
    }
    Variable res = {
        .val = res_val, 
        .deps = params,
        .opr = &Sum,
        .grad = 0.0
    };
    return res;
}

void _sum_backward(Variable* res) {
    VarArray deps = res->deps;
    if(deps.len == 0) return;

    for(size_t i=0; i<deps.len; i++) {
        _backward_grad(deps.data[i], res->grad);
    }
}

Operator Sum = {
    _sum_forward,
    _sum_backward
};


/*
* Sub
*/
Variable _sub_forward(VarArray params) {
    Variable** _params = params.data;
    float res_val = _params[0]->val;
    for(size_t i=1; i<params.len; i++) {
        res_val -= _params[i]->val;
    }
    Variable res = {
        .val = res_val, 
        .deps = params,
        .opr = &Sub,
        .grad = 0.0
    };
    return res;
}

void _sub_backward(Variable* res) {
    VarArray deps = res->deps;
    if(deps.len == 0) return;

    _backward_grad(deps.data[0], res->grad);
    for(size_t i=1; i<deps.len; i++) {
        _backward_grad(deps.data[i], -res->grad);
    }
}

Operator Sub = {
    _sub_forward,
    _sub_backward
};


/*
* Prod
*/
Variable _prod_forward(VarArray params) {
    Variable** _params = params.data;
    float res_val = 1.;
    for(size_t i=0; i<params.len; i++) {
        res_val *= _params[i]->val;
    }
    Variable res = {
        .val = res_val, 
        .deps = params,
        .opr = &Prod,
        .grad = 0.0
    };
    return res;
}

void _prod_backward(Variable* res) {
    VarArray deps = res->deps;
    if(deps.len == 0) return;

    for(size_t i=0; i<deps.len; i++) {
        float grad = 1.;
        for(size_t j=0; j<deps.len; j++) {
            if(i == j) continue;
            grad *= deps.data[j]->val;
        }
        _backward_grad(deps.data[i], grad * res->grad);
    }
}

Operator Prod = {
    _prod_forward,
    _prod_backward
};


/*
* Div
* binary opr
*/
Variable _div_forward(VarArray params) {
    Variable** _params = params.data;
    float res_val = _params[0]->val / _params[1]->val;
    Variable res = {
        .val = res_val, 
        .deps = params,
        .opr = &Div,
        .grad = 0.0
    };
    return res;
}

void _div_backward(Variable* res) {
    VarArray deps = res->deps;
    if(deps.len == 0) return;
    float left = deps.data[0]->val;
    float right = deps.data[1]->val;
    _backward_grad(deps.data[0], res->grad / right);
    _backward_grad(deps.data[1], - res->grad * left / (right * right) );
}

Operator Div = {
    _div_forward,
    _div_backward
};
