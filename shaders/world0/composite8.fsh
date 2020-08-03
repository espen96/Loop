#version 120
//Temporal Anti-Aliasing + Dynamic exposure calculations (vertex shader)

#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
#include "/lib/res_params.glsl"

#define fsh

#include "/program/comp/taa.glsl"