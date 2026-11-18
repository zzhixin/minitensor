#include <stdio.h>
#include "tensor.h"
#include <assert.h>
#include <string.h>
#include "random.h"

void test_vec_emul()
{
    Vec *u = mt_create_vec_by(5, (double[]){-9, 1, -2, -4, -6});
    Vec *v = mt_create_vec_by(5, (double[]){-2, 2, -2, -3, 5});
    Vec *w = mt_vec_emul(u, v);
    double gt[] = {18, 2, 4, 12, -30};
    assert(memcmp(w->data, gt, sizeof(gt)) == 0);

    Vec *z = mt_create_vec(5);
    mt_vec_emul_ip(z, u, v);
    assert(memcmp(z->data, gt, sizeof(gt)) == 0);
}

void test_vec_dot()
{
    Vec *u = mt_create_vec_by(5, (double[]){-9, 1, -2, -4, -6});
    Vec *v = mt_create_vec_by(5, (double[]){-2, 2, -2, -3, 5});
    double prod = mt_vec_dot(u, v);
    double gt = 6.;
    assert(gt == prod);
}

void test_mat_mul()
{
    Mat *A = mt_create_mat_by(2, 2, (double[]){1., 2., 3., 4.});
    Mat *B = mt_create_mat_by(2, 2, (double[]){1., 2., 3., 4.});
    double gt[] = {7., 10., 15., 22.};

    Mat *C = mt_create_mat(2, 2);
    mt_mat_mul_ip(C, A, B);
    assert(memcmp(C->data, gt, sizeof(gt)) == 0);
    // printf("A = \n");
    // mt_print_mat(A);
    // printf("\nB = \n");
    // mt_print_mat(B);
    // printf("\nC = A @ B = \n");
    // mt_print_mat(C);

    Mat *D = mt_mat_mul(A, B);
    assert(memcmp(D->data, gt, sizeof(gt)) == 0);
    // printf("\nD = A @ B = \n");
    // mt_print_mat(D);

    Mat *A1 = mt_create_mat_by(4, 5,
                               (double[]){-4, -3, -5, -3, -8, 8, 1, -1, -9, -8, -9, 1, 6, -5, 0, 2, -10, -9, -9, -2});
    Mat *B1 = mt_create_mat_by(5, 6,
                               (double[]){8, -7, 7, 7, -5, 10, 8, 0, 10, -6, -5, -4, 1, 9, -6, 4, -6, -10, -1, -1, -8, -7, -8, -1, 9, -1, -4, -10, 10, 5});
    double gt1[] = {-130, -6, 28, 71, 9, -15, 8, -48, 176, 189, -47, 55, -53, 122, -49, -10, 44, -149, -82, -84, 48, 121, 146, 149};

    Mat *C1 = mt_create_mat(4, 6);
    mt_mat_mul_ip(C1, A1, B1);
    assert(memcmp(C1->data, gt1, sizeof(gt1)) == 0);
    // printf("A1 = \n");
    // mt_print_mat(A1);
    // printf("\nB1 = \n");
    // mt_print_mat(B1);
    // printf("\nC1 = A1 @ B1 = \n");
    // mt_print_mat(C1);

    Mat *D1 = mt_mat_mul(A1, B1);
    assert(memcmp(D1->data, gt1, sizeof(gt1)) == 0);
    // printf("\nD1 = A1 @ B1 = \n");
    // mt_print_mat(D1);
}

void test_mat_mul_vec()
{
    Mat *A = mt_create_mat_by(5, 4, (double[]){3, -7, -10, -4, 5, 5, -4, 5, -4, -5, -5, 4, 8, -1, 5, -9, -8, 3, -7, -10});
    Vec *u = mt_create_vec_by(4, (double[]){-10, 9, -9, 10});
    double gt[] = {-43, 81, 80, -224, 70};

    Vec *v = mt_create_vec(5);
    mt_mat_mul_vec_ip(v, A, u);
    assert(memcmp(v->data, gt, sizeof(gt)) == 0);

    Vec *w = mt_mat_mul_vec(A, u);
    assert(memcmp(w->data, gt, sizeof(gt)) == 0);
}

int main()
{
    printf("> test_tensor: start test\n");
    test_vec_emul();
    test_vec_dot();
    test_mat_mul();
    test_mat_mul_vec();
    printf("  test_tensor: all tests passed!\n");
}