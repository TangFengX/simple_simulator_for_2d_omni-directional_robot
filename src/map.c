#include "map.h"
#include <math.h>
RayDirection ray_direction;
void map_init()
{
    // init map
    uint8_t i;
    for (i = 0; i < MAP_POINTS; i++)
    {
        map.points[i].x = 0;
        map.points[i].y = 0;
    }
    map.space_len = MAP_POINTS;
    map.head = 0;
    map.tail = 0;
    map.len = 0;
    map.time=0;
    // init ray direction cos/sin
    for (i = 0; i < RANGER_SAMPLE; i++)
    {
        ray_direction.cos_theta[i] = 0;
        ray_direction.sin_theta[i] = 0;
    }
    // init ranger date
    for (i = 0; i < RANGER_SAMPLE; i++)
    {
        ranger_data.dist[i] = 0;
        ranger_data.valid[i] = 0;
    }
    ranger_data.len = RANGER_SAMPLE;
    ranger_data.time=0;
    // init target list
    for (i = 0; i < MAP_TARGETS; i++)
    {
        target.x[i] = (float[])MAP_TARGETS_X[i];
        target.y[i] = (float[])MAP_TARGETS_Y[i];
    }
    target.p = 0;
}

void map_add_point(Point p)
{
    map.points[map.tail] = p;
    map.tail = (map.tail + 1) % map.space_len;
    if (map.len == map.space_len)
        map.head = map.tail;
    else
        map.len++;
}

void map_get_len(uint32_t *len)
{
    *len = map.len;
}

void map_get_point(uint32_t index, Point *p)
{
    if (index >= map.len)
        return;
    uint32_t idx = (map.head + index) % map.space_len;
    *p = map.points[idx];
}

void ray_direction_update()
{
    uint8_t i = RANGER_SAMPLE - 1;
    ray_direction.cos_theta[i] = cosf(imu.theta - RANGER_FOV / 2);
    ray_direction.sin_theta[i] = sinf(imu.theta - RANGER_FOV / 2);
    while (i > 0)
    {
        i--;
        ray_direction.cos_theta[i] = ray_direction.cos_theta[i + 1] * RANGER_RAY_ANGLE_COS - ray_direction.sin_theta[i + 1] * RANGER_RAY_ANGLE_SIN;
        ray_direction.sin_theta[i] = ray_direction.sin_theta[i + 1] * RANGER_RAY_ANGLE_COS + ray_direction.cos_theta[i + 1] * RANGER_RAY_ANGLE_SIN;
    }
}

void map_add_point_scanned()
{
    uint8_t i;
    Point p;
    static float last_x[RANGER_SAMPLE] = {};
    static float last_y[RANGER_SAMPLE] = {};
    static bool has_last[RANGER_SAMPLE] = {0};
    for (i = 0; i < RANGER_SAMPLE; i++)
    {
        if (!ranger_data.valid[i])
            continue;
        if (ranger_data.dist[i] < RANGER_MIN_DIST || ranger_data.dist[i] > RANGER_MAX_DIST)
        {
            continue;
        }

        p.x = imu.x + ray_direction.cos_theta[i] * ranger_data.dist[i];
        p.y = imu.y + ray_direction.sin_theta[i] * ranger_data.dist[i];
        if ((p.x - last_x[i]) * (p.x - last_x[i]) + (p.y - last_y[i]) * (p.y - last_y[i]) > MAP_VALID_POINT_MIN_DIST * MAP_VALID_POINT_MIN_DIST || !has_last[i])
        {
            last_x[i] = p.x;
            last_y[i] = p.y;
            map_add_point(p);
            has_last[i] = true;
        }
    }
}

void map_check_reach_target()
{
    if ((imu.x - target.x[target.p]) * (imu.x - target.x[target.p]) + (imu.y - target.y[target.p]) * (imu.y - target.y[target.p]) < MAP_TARGET_RADIUS * MAP_TARGET_RADIUS)
    {
        target.p = (target.p + 1) % MAP_TARGETS;
    }
}

void map_update()
{
    ray_direction_update();
    map_add_point_scanned();
    map_check_reach_target();
}

