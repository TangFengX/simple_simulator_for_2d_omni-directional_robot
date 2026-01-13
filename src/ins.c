#include "ins.h"
#include "pilot.h"

#define N POS_VECTOR_LEN
static float X[N];              // 状态
static float P_data[N*N];
static float F_data[N*N];
static float Q_data[N*N];
static Mat P, F, Q;

void kf_init()
{
    static uint8_t inited = 0;
    if (inited)
        return;
    inited = 1;

    init_mat(&P, P_data, N, N);
    init_mat(&F, F_data, N, N);
    init_mat(&Q, Q_data, N, N);
    float P0[N] = P0_DEFAULT;
    float Q0[N] = Q0_DEFAULT;
    for (int i = 0; i < N; i++)
    {
        X[i] = 0;
        for (int j = 0; j < N; j++)
        {
            P_data[i * N + j] = (i == j) ? P0[i] : 0;
            Q_data[i * N + j] = (i == j) ? Q0[i] : 0;
        }
    }
    imu_data_init();
}

void imu_data_init(){
    imu_data.time=0;
    imu_data.ah=0;
    imu_data.al=0;
    imu_data.omiga=0;
    imu_data.updated=false;
}


void kf_ins()
{
    kf_init();

    static float last_time = 0;
    float dt = imu_data.time - last_time;
    if (dt <= 0 || dt > 0.1f)
        dt = 0.01f;
    last_time = imu_data.time;

    float theta = X[2];
    float al = imu_data.al;
    float ah = imu_data.ah;

    /* ===== 1. 非线性状态预测 ===== */
    float ax = cosf(theta) * al - sinf(theta) * ah;
    float ay = sinf(theta) * al + cosf(theta) * ah;

    X[0] += X[3] * dt;
    X[1] += X[4] * dt;
    X[2] += imu_data.omiga * dt;
    X[3] += ax * dt;
    X[4] += ay * dt;

    /* ===== 2. 构建 F ===== */
    for (int i = 0; i < N * N; i++)
        F_data[i] = 0;

    F_data[0 * N + 0] = 1;
    F_data[0 * N + 3] = dt;
    F_data[1 * N + 1] = 1;
    F_data[1 * N + 4] = dt;
    F_data[2 * N + 2] = 1;
    F_data[3 * N + 2] = (-sinf(theta) * al - cosf(theta) * ah) * dt;
    F_data[3 * N + 3] = 1;
    F_data[4 * N + 2] = (cosf(theta) * al - sinf(theta) * ah) * dt;
    F_data[4 * N + 4] = 1;

    /* ===== 3. P = F P F^T + Q ===== */
    static float FP_data[N * N];
    static float FPFt_data[N * N];
    static float Ft_data[N * N];

    Mat FP, FPFt, Ft;
    init_mat(&FP, FP_data, N, N);
    init_mat(&FPFt, FPFt_data, N, N);
    init_mat(&Ft, Ft_data, N, N);

    mat_mul(&F, &P, &FP);
    mat_trans(&F, &Ft);
    mat_mul(&FP, &Ft, &FPFt);
    mat_add(&FPFt, &Q, &P);

    /* ===== 4. 输出到 imu ===== */
    imu.time = imu_data.time;
    imu.x = X[0];
    imu.y = X[1];
    imu.theta = X[2];
    imu.vx = X[3];
    imu.vy = X[4];
    imu.omiga = imu_data.omiga;
    imu.ax = ax;
    imu.ay = ay;
}
