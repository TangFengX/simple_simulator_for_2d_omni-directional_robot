#ifndef __INS_H__
#define __INS_H__
#include "pilot.h"
#include "math_utils.h"
// 惯性导航，使用卡尔曼滤波
void kf_ins();
//初始化所有矩阵
void kf_init();
#define N POS_VECTOR_LEN
static float X[N];              // 状态
static float P_data[N*N];
static float F_data[N*N];
static float Q_data[N*N];
static Mat P, F, Q;









#endif