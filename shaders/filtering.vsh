
#extension GL_EXT_gpu_shader4 : enable


out vec2 texcoord;

flat out vec3 WsunVec;

flat out vec3 ambientUp;
flat out vec3 ambientLeft;
flat out vec3 ambientRight;
flat out vec3 ambientB;
flat out vec3 ambientF;
flat out vec3 ambientDown;
flat out vec4 lightCol;
flat out float tempOffsets;
flat out vec2 TAA_Offset;
flat out vec3 zMults;
uniform sampler2D colortex4;
uniform float far;
uniform float near;
uniform mat4 gbufferModelViewInverse;
uniform vec3 sunPosition;
uniform float rainStrength;
uniform float sunElevation;
uniform int frameCounter;
flat out vec3 refractedSunVec;
uniform vec2 texelSize;
uniform int framemod8;

#include "/lib/kernel.glsl"


#include "/lib/util.glsl"
#include "/lib/res_params.glsl"
void main() {

	zMults = vec3(1.0/(far * near),far+near,far-near);
	gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0);
//	gl_Position.xy = (gl_Position.xy*0.5+0.5)*(0.01+VL_RENDER_RESOLUTION)*2.0-1.0;
	#ifdef TAA_UPSCALING
		gl_Position.xy = (gl_Position.xy*0.5+0.5)*RENDER_SCALE*2.0-1.0;
	#endif

	tempOffsets = HaltonSeq2(frameCounter%10000);
	TAA_Offset = offsets[frameCounter%8];
	#ifndef TAA
	TAA_Offset = vec2(0.0);
	#endif
    texcoord = gl_MultiTexCoord0.xy;



	

}
