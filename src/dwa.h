#ifndef __DWA_H__
#define __DWA_H__
typedef struct
{
    float max_v;
    float max_omiga;
    float max_a;
    float max_alpha;
    float dt;
    float predict_time;
    float w_goal;
    float w_obstacle;
    float w_velocity;
}DWAParam;
DWAParam dwa_param;
typedef struct {
    float x, y, theta;
    float v, omiga;
} SimState;


void dwa_param_init();
void simulate_step(SimState*s,float v, float omiga, float dt);
float score_goal(SimState*s);
float score_velocity(SimState*s,float v);
float score_obstacle(SimState*s);
void dwa_plan();
#endif // !__DWA_H__
