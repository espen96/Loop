#version 120
//downsample 1st pass (half res) for bloom
#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"
#define fsh

#include "/program/comp/bloom1.glsl"


