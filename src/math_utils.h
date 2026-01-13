#ifndef __MATH_UTILS__
#define __MATH_UTILS__
#include <math.h>
#include <stdint.h>
typedef struct{
    float*data;
    uint8_t row;
    uint8_t col;
}Mat;

//初始化矩阵
void init_mat(Mat*data,float*space,uint8_t row,uint8_t col);
//矩阵乘 a×b=c
void mat_mul(Mat*a,Mat*b,Mat*c);
//矩阵加 a+b=c
void mat_add(Mat*a,Mat*b,Mat*c);
//矩阵转置 a.T=b
void mat_trans(Mat*a,Mat*b);


#endif