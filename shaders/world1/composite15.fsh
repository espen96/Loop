#version 120
//Vignetting, applies bloom, applies exposure and tonemaps the final image
#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"
#define fsh
#define end
#include "/program/comp/dof.glsl"


