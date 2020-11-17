#version 120
//Merge and upsample the blurs into a 1/4 res bloom buffer

#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"
#define fsh

#include "/program/comp/bloom4.glsl"


