#version 150 compatibility
//Horizontal bilateral blur for volumetric fog + Forward rendered objects + Draw volumetric fog
#extension GL_EXT_gpu_shader4 : enable

#define NETHER
#include "/frend.fsh"