#ifndef __ARRAY_H__
#define  __ARRAY_H__

#include <stdint.h>
#include <stdbool.h>
#include "param.h"
#include "pilot.h"
//ranger data 需要的数组
uint16_t ranger_data_dist[RANGER_DATA_LEN];
bool ranger_data_valid[RANGER_DATA_LEN];

//map需要的数组
Point points[MAP_POINTS];


#endif