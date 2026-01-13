#ifndef __MAP_H__
#define __MAP_H__
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "pilot.h"
//更新地图(接收ranger数据进行扫描)
void map_update();
//初始化地图
void map_init();
//向地图中加入障碍物点
void map_add_point(Point p);
//读取地图中点的个数
void map_get_len(uint32_t*len);
//读取地图中第n个点
void map_get_point(uint32_t index,Point*p);

typedef struct{
    float cos_theta[RANGER_SAMPLE];
    float sin_theta[RANGER_SAMPLE];
}RayDirection;
RayDirection ray_direction;



//更新每一条ray的方向
void ray_direction_update();
//将扫描到的点加入到地图中
void map_add_point_scanned();
//检查是否到达终点
void map_check_reach_target();



#endif // !__MAP_H__
