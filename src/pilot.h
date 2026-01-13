#ifndef __PILOT_H__

#define __PILOT_H__

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "param.h"

// 来自tof的信息
typedef struct
{
    uint16_t dist[RANGER_SAMPLE];
    bool valid[RANGER_SAMPLE];
    uint8_t len;
} RangerData;
RangerData ranger_data;

// 来自imu的信息，包括前向，侧向加速度与角速度
typedef struct
{
    float time;
    float al;
    float ah;
    float omiga;
} ImuData;
ImuData imu_data;

// 传递命令给飞控
typedef struct
{
    float al;
    float ah;
    float alpha;
} Order;
Order order;

// 在二维下对自己的位姿估计
typedef struct
{
    float time;
    float x;
    float y;
    float theta;
    float vx;
    float vy;
    float omiga;
    float ax;
    float ay;
} Imu;
Imu imu;

// 点，用于后期的几何计算
typedef struct
{
    float x;
    float y;
} Point;


// 维护现有一定数量探测点的map
typedef struct
{
    Point points[MAP_POINTS];
    uint8_t space_len;
    uint8_t head;
    uint8_t tail;
    uint8_t len;
} Map;
Map map;

//存储一系列的航迹点，当判断机体到达一个航迹点时，就p++切换到下一个，循环滚动
typedef struct{
    float x[MAP_TARGETS];
    float y[MAP_TARGETS];
    uint8_t p;
}Target;
Target target;

// DWA规划器
void dwa_planner();

//调用各模块初始化函数，初始化各模块
void init_pilot();

//更新，做出决策
void update_pilot();

#endif