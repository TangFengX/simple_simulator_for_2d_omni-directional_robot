#include "pilot.h"
#include "array.h"

void ranger_data_init(RangerData*ranger_data){
    ranger_data->dist=ranger_data_dist;
    ranger_data->valid=ranger_data_valid;
    ranger_data->len=RANGER_DATA_LEN;
}

void map_init(Map*map){
    map->points=points;
    map->len=MAP_POINTS;
    map->head=0;
    map->tail=0;
}

 