#ifndef __INS_H__
#define __INS_H__
#include "pilot.h"
#include "math_utils.h"
// 惯性导航，使用卡尔曼滤波
void kf_ins();
//初始化所有矩阵
void kf_init();


void imu_data_init();







#endif