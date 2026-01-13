#include "math_utils.h"

void init_mat(Mat *m, float *space, uint8_t row, uint8_t col)
{
    m->data = space;
    m->row = row;
    m->col = col;
}

void mat_mul(Mat *a, Mat *b, Mat *c)
{
    uint8_t i, j, k;
    for (i = 0; i < a->row; i++)
    {
        for (j = 0; j < b->col; j++)
        {
            c->data[i * c->col + j] = 0.0f;
            for (k = 0; k < a->col; k++)
            {
                c->data[i * c->col + j] +=
                    a->data[i * a->col + k] *
                    b->data[k * b->col + j];
            }
        }
    }
}

void mat_add(Mat* a, Mat* b, Mat* c)
{
    uint16_t n = a->row * a->col;
    for(uint16_t i=0;i<n;i++)
        c->data[i] = a->data[i] + b->data[i];
}

void mat_trans(Mat* a, Mat* b)
{
    for(uint8_t i=0;i<a->row;i++)
        for(uint8_t j=0;j<a->col;j++)
            b->data[j*b->col + i] =
                a->data[i*a->col + j];
}