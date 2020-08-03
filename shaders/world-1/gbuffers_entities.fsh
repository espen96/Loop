#version 120
#extension GL_EXT_gpu_shader4 : enable
#extension GL_ARB_shader_texture_lod : enable

#define TEMPORARY_FIX
#define mask
#define entities
#include "/lib/settings.glsl"
#include "/program/gbuffers/solid.glsl"



