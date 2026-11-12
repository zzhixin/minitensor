#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "autograd.h"
#include <string.h>

static const double TIMING_RUNS = 1000;

double mt_rand_between(double l, double r)
{
    long int random_num = rand();
    double res = (double)random_num / (RAND_MAX + 1.) * (r - l) + l;
    return res;
}

int mt_rand_between_int(int l, int r)
{
    assert(r >= l);
    int x = round(mt_rand_between(l - 0.5, r + 0.5));
    if (x < l)
        return l;
    else if (x > r)
        return r;
    return x;
}

void mt_rand_between_diff_2_int(int *first, int *second, int l, int r)
{
    assert(r > l);
    *first = mt_rand_between_int(l, r);
    int diff = mt_rand_between_int(1, r - l);
    *second = (*first + diff) % (r - l + 1);
}

double mt_rand()
{
    return mt_rand_between(0., 1.);
}

void generate_complex_graph(Graph *graph, size_t graph_size, size_t num_leaf_node)
{
    assert(graph->size == 0);
    assert(graph_size <= graph->capcity);
    assert(num_leaf_node >= 2);
    // Summing all leaves requires at least num_leaf_node - 1 binary nodes.
    assert(num_leaf_node - 1 <= graph_size);
    assert(num_leaf_node <= graph_size - (num_leaf_node - 1));
    bool is_terminal[graph_size];
    memset(is_terminal, 0, sizeof(is_terminal));
    size_t terminal_count = num_leaf_node;
    for (size_t i = 0; i < num_leaf_node; i++)
    {
        mt_create_var(graph, mt_rand());
        is_terminal[i] = true;
    }

    size_t i = num_leaf_node;
    for (; i < graph_size; i++)
    {
        size_t remaining = graph_size - i;
        if (remaining == terminal_count - 1)
            break;

        // With one spare slot, extend the latest terminal without adding
        // another terminal, then use all remaining slots for the sum.
        if (remaining == terminal_count)
        {
            mt_sin(graph, &graph->data[i - 1]);
        }
        else
        {
            double opr_dice = mt_rand_between(0., 8.);
            if (opr_dice < 4.)
            { // binary opr
                Variable *(*opr)(Graph *, Variable *, Variable *);
                if (opr_dice < 1.)
                    opr = mt_add;
                else if (opr_dice < 2.)
                    opr = mt_sub;
                else if (opr_dice < 3.)
                    opr = mt_mul;
                else if (opr_dice < 4.)
                    // opr = mt_div;
                    opr = mt_mul; // div may cause nan.
                int deps_dice_left;
                int deps_dice_right;
                mt_rand_between_diff_2_int(&deps_dice_left, &deps_dice_right, 0, i - 1);
                Variable *left = &graph->data[deps_dice_left];
                Variable *right = &graph->data[deps_dice_right];
                opr(graph, left, right);
            }
            else
            {
                // unary opr
                Variable *(*opr)(Graph *, Variable *);
                if (opr_dice < 5.)
                    opr = mt_exp;
                else if (opr_dice < 6.)
                    opr = mt_log;
                else if (opr_dice < 7.)
                    opr = mt_sin;
                else if (opr_dice < 8.)
                    opr = mt_cos;
                int dep_dice = mt_rand_between_int(0, i - 1);
                Variable *dep = &graph->data[dep_dice];
                opr(graph, dep);
            }
        }

        is_terminal[i] = true;
        terminal_count++;
        for (size_t j = 0; j < graph->data[i].deps_size; j++)
        {
            size_t dep = (size_t)(graph->data[i].deps[j] - graph->data);
            if (is_terminal[dep])
            {
                is_terminal[dep] = false;
                terminal_count--;
            }
        }
    }

    // Every generated node is an ancestor of at least one terminal.
    // Reserve terminal_count - 1 slots to join them into a single output.
    size_t generated = i;
    Variable *sum = NULL;
    for (size_t j = 0; j < generated; j++)
    {
        if (!is_terminal[j])
            continue;
        if (sum == NULL)
        {
            sum = &graph->data[j];
        }
        else
        {
            sum = mt_add(graph, sum, &graph->data[j]);
            i++;
        }
    }
    assert(i == graph_size);
    assert(graph->size == graph_size);
    assert(sum == &graph->data[graph_size - 1]);
}

void test_generate_complex_graph()
{
    srand(time(NULL));

    int graph_size = 1000;
    int num_leaf = 10;
    Graph *graph = mt_create_graph(graph_size);
    generate_complex_graph(graph, graph_size, num_leaf);

    printf("add: %p\n", mt_add_backward);
    printf("sub: %p\n", mt_sub_backward);
    printf("mul: %p\n", mt_mul_backward);
    printf("div: %p\n", mt_div_backward);
    printf("exp: %p\n", mt_exp_backward);
    printf("log: %p\n", mt_log_backward);
    printf("sin: %p\n", mt_sin_backward);
    printf("cos: %p\n\n", mt_cos_backward);

    for (size_t i = 0; i < graph_size; i++)
    {
        Variable *node = &graph->data[i];
        printf("node: %p\n  backward: %p\n  deps[0]: %p\n  deps[1]: %p\n  val: %.2f\n",
               node, node->backward, node->deps[0], node->deps[1], node->val);
    }
    free(graph);
}

void time_complex_graph()
{
    srand(0); // Use the same graph in both versions.

    int graph_size = 1000;
    int num_leaf = 5;
    Graph *graph = mt_create_graph(graph_size);
    generate_complex_graph(graph, graph_size, num_leaf);

    clock_t start = clock();
    for (size_t i = 0; i < TIMING_RUNS; i++)
    {
        clear_graph_grads(graph);
        mt_backward(&graph->data[graph_size - 1]);
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC / TIMING_RUNS;
    printf("complex graph rebuild order time spent: %.4f us\n", time_spent * 1e6);

    start = clock();
    for (size_t i = 0; i < TIMING_RUNS; i++)
    {
        clear_graph_grads(graph);
        mt_backward_graph(graph);
    }
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC / TIMING_RUNS;
    printf("complex graph use graph order time spent: %.4f us\n", time_spent * 1e6);

    free(graph);
}

void time_simple_graph()
{
    Graph *graph = mt_create_graph(100);
    Variable *a = mt_create_var_debug(graph, 2., "a");
    Variable *b = mt_create_var_debug(graph, 3., "b");
    Variable *c = mt_sin(graph, a);
    strlcpy(c->debug, "c", sizeof(c->debug));
    Variable *d = mt_div(graph, c, b);
    strlcpy(d->debug, "d", sizeof(c->debug));
    Variable *e = mt_exp(graph, d);
    strlcpy(e->debug, "e", sizeof(c->debug));
    Variable *f = mt_add(graph, c, e);
    strlcpy(f->debug, "f", sizeof(c->debug));

    clock_t start = clock();
    for (size_t i = 0; i < TIMING_RUNS; i++)
    {
        clear_graph_grads(graph);
        mt_backward(f);
    }
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC / TIMING_RUNS;
    printf("simple graph rebuild order time spent: %.4f us\n", time_spent * 1e6);

    start = clock();
    for (size_t i = 0; i < TIMING_RUNS; i++)
    {
        clear_graph_grads(graph);
        mt_backward_graph(graph);
    }
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC / TIMING_RUNS;
    printf("simple graph use graph order time spent: %.4f us\n", time_spent * 1e6);
}

int main()
{
    time_simple_graph();
    time_complex_graph();
}