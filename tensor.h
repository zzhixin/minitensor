#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

typedef struct Tensor1D
{
    size_t len;
    double data[];
} Tensor1D;
typedef Tensor1D Vec;

typedef struct Tensor2D
{
    size_t row;
    size_t col;
    double data[];
} Tensor2D;
typedef Tensor2D Mat;

Vec *mt_create_vec_by(size_t len, double *data);
Vec *mt_create_vec(size_t len);
void mt_print_vec(Vec *);
void mt_vec_emul_ip(Vec *, Vec *, Vec *);
Vec *mt_vec_emul(Vec *, Vec *);
double mt_vec_dot(Vec *, Vec *);

Mat *mt_create_mat_by(size_t row, size_t col, double *data);
Mat *mt_create_mat(size_t row, size_t col);
Mat *mt_create_rand_mat(size_t row, size_t col, double l, double r);
void mt_print_mat(Mat *mat);
void mt_mat_mul_ip(Mat *C, Mat *A, Mat *B);
Mat *mt_mat_mul(Mat *A, Mat *B);
void mt_mat_mul_vec_ip(Vec *v, Mat *A, Vec *u);
Vec *mt_mat_mul_vec(Mat *A, Vec *u);

#endif