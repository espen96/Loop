#version 150
// Filter

#extension GL_EXT_gpu_shader4 : enable
#define DENOISE_RANGE 8
int FILTER_STAGE = 3;

#include "/filtering.fsh"
