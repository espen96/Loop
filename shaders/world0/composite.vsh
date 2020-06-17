#version 120
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
flat varying vec2 TAA_Offset;
flat varying float tempOffsets;
flat varying vec3 WsunVec;
uniform int frameCounter;
uniform float sunElevation;
uniform vec3 sunPosition;
uniform mat4 gbufferModelViewInverse;
#include "/lib/util.glsl"
const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
							vec2(-1.,3.)/8.,
							vec2(5.0,1.)/8.,
							vec2(-3,-5.)/8.,
							vec2(-5.,5.)/8.,
							vec2(-7.,-1.)/8.,
							vec2(3,7.)/8.,
							vec2(7.,-7.)/8.);
void main() {
	tempOffsets = frameCounter/1.6180339887;
	TAA_Offset = offsets[frameCounter%8];
	#ifndef TAA
	TAA_Offset = vec2(0.0);
	#endif
	gl_Position = ftransform();


	WsunVec = (float(sunElevation > 1e-5)*2-1.)*normalize(mat3(gbufferModelViewInverse) *sunPosition);

}
