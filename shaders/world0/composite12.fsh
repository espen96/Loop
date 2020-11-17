#version 120
//6 Horizontal gaussian blurs and horizontal downsampling

#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"
#define fsh

#include "/program/comp/bloom2.glsl"


