#ifndef __PARAM_H__
#define __PARAM_H__

//自动生成，请勿修改

#define MAP_POINTS                       64
#define MAP_VALID_POINT_MIN_DIST         0.2
#define MAP_TARGETS                      2
#define MAP_TARGETS_X                    {8, 0}
#define MAP_TARGETS_Y                    {8, 0}
#define MAP_TARGET_RADIUS                0.2
#define POS_VECTOR_LEN                   7 // [x y theta vx vy b_a前向0偏 b_g陀螺0偏]
#define P0_DEFAULT                       {1.0, 1.0, 0.2, 0.5, 0.5, 0.05, 0.01} // x y theta ax ay b_ax b_omiga
#define Q0_DEFAULT                       {1e-06, 1e-06, 1e-07, 0.001, 0.001, 1e-06, 1e-08}
#define RANGER_FOV                       1.0471975511965976
#define RANGER_SAMPLE                    8
#define RANGER_RAY_ANGLE_COS             0.98883082623 // cos(fov/(sample-1))
#define RANGER_RAY_ANGLE_SIN             0.14904226618 // sin(fov/(sample-1))
#define RANGER_MIN_DIST                  0.1
#define RANGER_MAX_DIST                  4
#define ISA_WORK_CYCLE                   0.01 // 100hz
#define DWA_WORK_CYCLE                   0.1 // 10hz
#define MAX_V                            2 // 最大速度 m/s
#define MAX_OMIGA                        3.14159265359 // 最大角速度 rad/s
#define MAX_A                            1 // 最大加速度 m/s^2
#define MAX_ALPHA                        2 // 最大角加速度 rad/s^2
#define DWA_DT                           0.1 // DWA预测步长
#define DWA_PREDICT_TIME                 1.5 // DWA预测时长
#define DWA_WEIGHT_GOAL                  1 // 目标点权重
#define DWA_WEIGHT_OBSTACLE              1.5 // 障碍物权重
#define DWA_WEIGHT_VELOCITY              0.3 // 速度权重
#define DWA_MIN_DISTANCE_TO_OBSTACLE     0.1 // 距离障碍物最小距离
#define DWA_VELOCITY_SEARCH_STEP         0.1 // 速度搜索步长
#define DWA_ANGULAR_VELOCITY_SEARCH_STEP 0.1 // 角速度搜索步长

#endif // !__PARAM_H__