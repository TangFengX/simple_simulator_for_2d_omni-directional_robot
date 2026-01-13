#include "dwa.h"
#include "param.h"
#include "pilot.h"
#include <math.h>
#include "math_utils.h"

// 预计算常量，避免重复计算
static float MAX_ACCEL_TIME_DT; // max_a * dt
static float MAX_ALPHA_TIME_DT; // max_alpha * dt
static float MIN_DIST_SQ;       // 安全距离的平方

void dwa_param_init()
{
    dwa_param.max_v = MAX_V;
    dwa_param.max_omiga = MAX_OMIGA;
    dwa_param.max_a = MAX_A;
    dwa_param.max_alpha = MAX_ALPHA;
    dwa_param.dt = DWA_DT;
    dwa_param.predict_time = DWA_PREDICT_TIME;
    dwa_param.w_goal = DWA_WEIGHT_GOAL;
    dwa_param.w_obstacle = DWA_WEIGHT_OBSTACLE;
    dwa_param.w_velocity = DWA_WEIGHT_VELOCITY;

    // 预计算优化
    MAX_ACCEL_TIME_DT = dwa_param.max_a * dwa_param.dt;
    MAX_ALPHA_TIME_DT = dwa_param.max_alpha * dwa_param.dt;
    float safe_dist = DWA_MIN_DISTANCE_TO_OBSTACLE;
    MIN_DIST_SQ = safe_dist * safe_dist;
}

// 辅助函数声明
static void simulate_trajectory(SimState *s, float v, float w, float total_time, float dt);
static float fast_score_obstacle(const SimState *s); // 优化后的障碍物评分

void dwa_plan()
{
    float best_score = -1e9f;
    float best_v = 0.0f; // 初始化，防止无解
    float best_w = 0.0f;
    float v_now = imu.vx * cosf(imu.theta) + imu.vy * sinf(imu.theta);
    float omiga = imu.omiga;
    float v_min = v_now - MAX_ACCEL_TIME_DT;
    float v_max = v_now + MAX_ACCEL_TIME_DT;
    float w_min = omiga - MAX_ALPHA_TIME_DT;
    float w_max = omiga + MAX_ALPHA_TIME_DT;

    // 根据绝对速度限制裁切
    v_min = fmaxf(-dwa_param.max_v, v_min);
    v_max = fminf(dwa_param.max_v, v_max);
    w_min = fmaxf(-dwa_param.max_omiga, w_min);
    w_max = fminf(dwa_param.max_omiga, w_max);

    // 性能优化提示：如果算力依然吃紧，可以增大 SEARCH_STEP
    SimState s_final;

    for (float v = v_min; v <= v_max; v += DWA_VELOCITY_SEARCH_STEP)
    {
        for (float w = w_min; w <= w_max; w += DWA_ANGULAR_VELOCITY_SEARCH_STEP)
        {
            // 重置模拟状态到当前位置
            s_final.x = imu.x;
            s_final.y = imu.y;
            s_final.theta = imu.theta;

            // 预测未来轨迹 (直接预测到终点)
            simulate_trajectory(&s_final, v, w, dwa_param.predict_time, dwa_param.dt);

            // 4. 评分计算
            // 优先计算代价小的项。如果速度项权重很大且当前速度很小，或许可以剪枝(通常不这么做)
            float score_obs = score_obstacle(&s_final);

            // 如果撞了，直接跳过后续昂贵的 atan2 计算
            if (score_obs < -1e5f)
                continue;

            float score_g = score_goal(&s_final);
            float score_v = score_velocity(&s_final, v);

            float total_score = dwa_param.w_goal * score_g +
                                dwa_param.w_obstacle * score_obs +
                                dwa_param.w_velocity * score_v;

            if (total_score > best_score)
            {
                best_score = total_score;
                best_v = v;
                best_w = w;
            }
        }
    }

    if (best_score < -1e4f)
    {
        best_v = 0;
        best_w = omiga > 0 ? dwa_param.max_omiga/2 : -dwa_param.max_omiga/2;
    }
    float a_cmd = (best_v - v_now) / dwa_param.dt;

    if (a_cmd > dwa_param.max_a)
        a_cmd = dwa_param.max_a;
    else if (a_cmd < -dwa_param.max_a)
        a_cmd = -dwa_param.max_a;

    order.al = a_cmd;
    order.ah = 0.0f; // 高度保持或另外控制

    // 角加速度
    float alpha_cmd = (best_w - omiga) / dwa_param.dt;
    if (alpha_cmd > dwa_param.max_alpha)
        alpha_cmd = dwa_param.max_alpha;
    else if (alpha_cmd < -dwa_param.max_alpha)
        alpha_cmd = -dwa_param.max_alpha;

    order.alpha = alpha_cmd;
}

static void simulate_trajectory(SimState *s, float v, float w, float total_time, float dt)
{
    // 简单欧拉积分。如果在低速下，可以将多次小步长合并为一次大步长运算
    // 或者直接使用运动学公式：
    // x(t) = x0 + v/w * (sin(theta0 + w*t) - sin(theta0))
    // 但考虑到除零风险(w=0)和计算量(sin/cos)，循环累加在MCU上往往更快且通用。

    for (float t = 0; t < total_time; t += dt)
    {
        s->x += v * cosf(s->theta) * dt;
        s->y += v * sinf(s->theta) * dt;
        s->theta += w * dt;
    }
}

// 优化后的障碍物评分函数
float score_obstacle(SimState *s)
{
    float min_dist_sq = 1e9f; // 记录最小距离的平方

    // 遍历所有障碍物点
    for (int i = 0; i < map.len; i++)
    {
        float dx = map.points[i].x - s->x;
        float dy = map.points[i].y - s->y;

        // 只算平方，不开方
        float d2 = dx * dx + dy * dy;

        // 【关键优化1】: 碰撞检测提前退出 (Early Exit)
        // 一旦发现任意一个点小于安全距离，立刻判定为不可行
        if (d2 < MIN_DIST_SQ)
        {
            return -1e6f; // 极大的惩罚，代表碰撞
        }

        if (d2 < min_dist_sq)
            min_dist_sq = d2;
    }

    // 计算分数
    // 使用平方根倒数或其他衰减函数。
    // 如果 min_dist_sq 很大，说明很安全。
    // 为了节省 sqrtf，我们可以直接用平方来做分数映射，只要单调性一致即可
    // 原逻辑：score = 1 - 1/(d+1) => d越大 score越大

    // 这里保留开方以维持你原有的物理量级感，但在MCU上如果极其严苛，可用 d2 代替
    float d = sqrtf(min_dist_sq);

    // 简单的归一化逻辑
    float score = 1.0f - 1.0f / (d + 1.0f);
    return score;
}

float score_velocity(SimState *s, float v)
{
    // 保持原样，逻辑简单且正确
    // 倾向于向前走(v>0)，惩罚倒车
    return fabsf(v) / dwa_param.max_v * (v >= 0 ? 1.0f : 0.5f);
}

float score_goal(SimState *s)
{
    // 计算目标向量
    float dx = target.x[target.p] - s->x;
    float dy = target.y[target.p] - s->y;

    // atan2 比较耗时，但在外层循环中无法避免，除非用近似算法
    float goal_theta = atan2f(dy, dx);
    float dtheta = goal_theta - s->theta;

    // 角度归一化 (-PI, PI)
    // 使用 math_utils 中的 helper 或者手动逻辑
    while (dtheta > 3.14159265f)
        dtheta -= 6.2831853f;
    while (dtheta < -3.14159265f)
        dtheta += 6.2831853f;

    // 评分：角度差越小分越高
    return -fabsf(dtheta) / 3.14159265f; // 结果范围 [-1, 0]
}