#version 120
//downsample 1st pass (half res) for bloom
#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"
#define vsh

#include "/program/comp/bloom0.glsl"


