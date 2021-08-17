#version 150
//Filter 

#extension GL_EXT_gpu_shader4 : enable
#define DENOISE_RANGE 4
int FILTER_STAGE = 2;


#include "/filtering.fsh"
