#ifndef __PILOT_H__

#define __PILOT_H__

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "array.h"

// 来自tof的信息
typedef struct
{
    uint16_t *dist;
    bool *valid;
    uint8_t len;
} RangerData;

// 初始化唯一的RangerData结构体
void ranger_data_init(
    RangerData *ranger_data);
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
    float omiga;
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
    float alpha;
} Imu;

Imu imu;

// 点，用于后期的几何计算
typedef struct
{
    float x;
    float y;
} Point;

Point Target;

// 维护现有一定数量探测点的map
typedef struct
{
    Point *points;
    uint32_t len;
    uint32_t head;
    uint32_t tail;
} Map;

Map map;

// 初始化map
void map_init(
    Map *map);

// DWA规划器
void dwa_planner(
    RangerData *ranger_data,
    ImuData *imu_data,
    Order *order,
    Imu *imu,
    Point *Target);

void mapping(
    RangerData *ranger_data,
    ImuData *imu_data,
    Imu *imu,
    Map *map);

// 惯性导航，使用卡尔曼滤波
void kf_ins(
    ImuData *imu_data,
    Imu *imu);
#endif