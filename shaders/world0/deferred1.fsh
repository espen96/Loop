#version 150
#extension GL_EXT_gpu_shader4 : enable

//Computes volumetric clouds at variable resolution (default 1/4 res)
#define HQ_CLOUDS	//Renders detailled clouds for viewport
#define CLOUDS_QUALITY 0.35 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define TAA
#define VOLUMETRIC_CLOUDS

flat in vec3 sunColor;
flat in vec3 moonColor;
flat in vec3 avgAmbient;
flat in float tempOffsets;

uniform sampler2D depthtex0;
uniform sampler2D noisetex;
uniform sampler2D colortex4;
uniform sampler2D colortex6;
uniform sampler2D colortex13;
uniform sampler2D colortex15;
uniform sampler2D colortex2;

uniform vec3 sunVec;
uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform int frameCounter;
uniform int framemod8;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform vec3 cameraPosition;

vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}

#include "/lib/sky_gradient.glsl"
#include "/lib/util.glsl"
#include "/lib/noise.glsl"


#ifdef VOLUMETRIC_CLOUDS
#include "/lib/volumetricClouds.glsl"
#endif
#include "/lib/res_params.glsl"
#include "/lib/kernel.glsl"

//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
	float checkerboard(in vec2 uv)
{
    vec2 pos = floor(uv);
  	return mod(pos.x + mod(pos.y, 2.0), 2.0);
}		


void main() {

/* RENDERTARGETS: 0 */


//  float checker =checkerboard(gl_FragCoord.xy);

	#ifdef VOLUMETRIC_CLOUDS
	vec2 halfResTC = vec2(floor(gl_FragCoord.xy)/CLOUDS_QUALITY/RENDER_SCALE+0.5+offsets[framemod8]*CLOUDS_QUALITY*RENDER_SCALE*0.5);

	vec3 fragpos = toScreenSpace(vec3(halfResTC*texelSize,1.0));
	vec4 currentClouds = vec4(0.0);

//	if(checker <0.5)	currentClouds = renderClouds(fragpos,vec3(0.), blueNoise(),sunColor/150.,moonColor/150.,avgAmbient/150.);
	currentClouds = renderClouds(fragpos,vec3(0.), R2_dither(),sunColor/150.,moonColor/150.,avgAmbient/150.);
	
	gl_FragData[0] = currentClouds;


	#else
		gl_FragData[0] = vec4(0.0,0.0,0.0,1.0);
	#endif

}
