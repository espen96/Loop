#version 120
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
const bool shadowHardwareFiltering = true;

#define fsh
#define NETHER

#include "/program/comp/lighting1.glsl"
