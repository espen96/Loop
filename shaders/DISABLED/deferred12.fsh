#version 140
//Filter 

#extension GL_EXT_gpu_shader4 : enable
#define DENOISE_RANGE 16
int FILTER_STAGE = 4;

#include "/filtering.fsh"
