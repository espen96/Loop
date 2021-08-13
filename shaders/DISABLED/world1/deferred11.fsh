#version 140
//Filter 

#extension GL_EXT_gpu_shader4 : enable
#define DENOISE_RANGE 32
int FILTER_STAGE = 5;

#include "/filtering.fsh"
