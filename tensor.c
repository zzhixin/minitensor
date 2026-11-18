#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "random.h"

Vec *mt_create_vec_by(size_t len, double *data)
{
    size_t data_size = len * sizeof(double);
    Vec *v = malloc(sizeof(Vec) + data_size);
    v->len = len;
    memcpy(v->data, data, data_size);
    return v;
}

Vec *mt_create_vec(size_t len)
{
    size_t data_size = len * sizeof(double);
    Vec *v = calloc(1, sizeof(Vec) + data_size);
    v->len = len;
    return v;
}

void mt_print_vec(Vec *v)
{
    size_t len = v->len;
    for (size_t i = 0; i < len; i++)
        printf(" %8.3f", v->data[i]);
    printf("\n");
}

void mt_vec_emul_ip(Vec *w, Vec *u, Vec *v)
{
    assert(w->len == u->len && u->len == v->len);
    size_t len = w->len;
    for (size_t i = 0; i < len; i++)
    {
        w->data[i] = u->data[i] * v->data[i];
    }
}

Vec *mt_vec_emul(Vec *u, Vec *v)
{
    assert(u->len == v->len);
    size_t len = u->len;
    Vec *w = mt_create_vec(len);
    for (size_t i = 0; i < len; i++)
        w->data[i] = u->data[i] * v->data[i];
    return w;
}

double mt_vec_dot(Vec *u, Vec *v)
{
    assert(u->len == v->len);
    size_t len = u->len;
    double res = 0.;
    for (size_t i = 0; i < len; i++)
        res += u->data[i] * v->data[i];
    return res;
}

Mat *mt_create_mat_by(size_t row, size_t col, double *data)
{
    // create matrix on heap
    size_t data_size = row * col * sizeof(double);
    Mat *mat = malloc(sizeof(Mat) + data_size);
    mat->row = row;
    mat->col = col;
    memcpy(mat->data, data, data_size);
    return mat;
}

Mat *mt_create_mat(size_t row, size_t col)
{
    // create matrix on heap
    size_t data_size = row * col * sizeof(double);
    Mat *mat = calloc(1, sizeof(Mat) + data_size);
    mat->row = row;
    mat->col = col;
    return mat;
}

Mat *mt_create_rand_mat(size_t row, size_t col, double l, double r)
{
    size_t data_size = row * col * sizeof(double);
    Mat *mat = malloc(sizeof(Mat) + data_size);
    mat->row = row;
    mat->col = col;
    for (size_t i = 0; i < row; i++)
        for (size_t j = 0; j < col; j++)
            mat->data[i * col + j] = mt_rand_between(l, r);
    return mat;
}

void mt_print_mat(Mat *mat)
{
    size_t m = mat->row;
    size_t n = mat->col;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
            printf(" %8.3f", mat->data[i * n + j]);
        printf("\n");
    }
}

void mt_mat_mul_ip(Mat *C, Mat *A, Mat *B)
{
    size_t m = A->row;
    size_t p = A->col;
    size_t n = B->col;
    memset(C->data, 0, m * n * sizeof(double));
    for (size_t i = 0; i < m; i++)
        for (size_t k = 0; k < p; k++)
            for (size_t j = 0; j < n; j++)
                C->data[i * n + j] += A->data[i * p + k] * B->data[k * n + j];
}

Mat *mt_mat_mul(Mat *A, Mat *B)
{
    size_t m = A->row;
    size_t p = A->col;
    size_t n = B->col;
    Mat *C = mt_create_mat(m, n);
    for (size_t i = 0; i < m; i++)
        for (size_t k = 0; k < p; k++)
            for (size_t j = 0; j < n; j++)
                C->data[i * n + j] += A->data[i * p + k] * B->data[k * n + j];
    return C;
}

void mt_mat_mul_vec_ip(Vec *v, Mat *A, Vec *u)
{
    size_t m = A->row;
    size_t n = A->col;
    memset(v->data, 0, n * sizeof(double));
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            v->data[i] += A->data[i * n + j] * u->data[j];
}

Vec *mt_mat_mul_vec(Mat *A, Vec *u)
{
    size_t m = A->row;
    size_t n = A->col;
    Vec *v = mt_create_vec(m);
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            v->data[i] += A->data[i * n + j] * u->data[j];
    return v;
}