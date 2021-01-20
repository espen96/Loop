#version 120
#extension GL_EXT_gpu_shader4 : enable


#define WAVY_PLANTS
#define WAVY_STRENGTH 1.0 //[0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0]
#define WAVY_SPEED 1.0 //[0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.25 1.5 2.0 3.0 4.0]
#define SEPARATE_AO
//#define POM
//#define USE_LUMINANCE_AS_HEIGHTMAP	//Can generate POM on any texturepack (may look weird in some cases)

#ifndef USE_LUMINANCE_AS_HEIGHTMAP
#ifndef MC_NORMAL_MAP
#undef POM
#endif
#endif

#ifdef POM
#define MC_NORMAL_MAP
#endif




#define textured
#define solid1
#include "/lib/res_params.glsl"
#include "/program/gbuffer_vertex.vfs"