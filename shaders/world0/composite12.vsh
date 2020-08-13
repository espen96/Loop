#version 120
//6 Vertical gaussian blurs and vertical downsampling

#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"
#define vsh

#include "/program/comp/bloom3.glsl"


