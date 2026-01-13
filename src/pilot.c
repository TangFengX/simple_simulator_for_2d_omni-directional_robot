#include "pilot.h"
#include "ins.h"
#include "map.h"
#include "dwa.h"

RangerData ranger_data;
ImuData imu_data;
Order order;
Imu imu;
Map map;
Target target;
void pilot_init()
{
   kf_init();
   map_init();
   dwa_init();
}

void pilot_update(float t)
{
   if(ranger_data.updated==true){
      ranger_data.updated=false;
      if(ranger_data.time-map.time>MAP_UPDATE_WORK_CYCLE){
         map_update();
         map.time=ranger_data.time;
      }
   }
   if(imu_data.updated==true){
      imu_data.updated=false;
      if(imu_data.time-imu.time>INS_WORK_CYCLE){
         kf_ins();
         imu.time=imu_data.time;
      }
   }
   if(t-order.time>DWA_WORK_CYCLE){
      dwa_plan();
      order.time=t;
   }
}