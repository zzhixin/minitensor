#ifndef RANDOM_H
#define RANDOM_H

#include <stdlib.h>
#include <assert.h>
#include <math.h>

static inline double mt_rand_between(double l, double r)
{
    long int random_num = rand();
    double res = (double)random_num / (RAND_MAX + 1.) * (r - l) + l;
    return res;
}

static inline int mt_rand_between_int(int l, int r)
{
    assert(r >= l);
    int x = round(mt_rand_between(l - 0.5, r + 0.5));
    if (x < l)
        return l;
    else if (x > r)
        return r;
    return x;
}

static inline void mt_rand_between_diff_2_int(int *first, int *second, int l, int r)
{
    assert(r > l);
    *first = mt_rand_between_int(l, r);
    int diff = mt_rand_between_int(1, r - l);
    *second = (*first + diff) % (r - l + 1);
}

static inline double mt_rand()
{
    return mt_rand_between(0., 1.);
}

#endif