
varying vec2 texcoord;

flat varying vec3 WsunVec;

flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;
flat varying vec4 lightCol;
flat varying float tempOffsets;
flat varying vec2 TAA_Offset;
flat varying vec3 zMults;
attribute vec4 mc_Entity;
uniform sampler2D colortex4;
uniform float far;
uniform float near;
uniform mat4 gbufferModelViewInverse;
uniform vec3 sunPosition;
uniform float rainStrength;
uniform float sunElevation;
uniform int frameCounter;
flat varying vec3 refractedSunVec;
uniform vec2 texelSize;
uniform int framemod8;

const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
							vec2(-1.,3.)/8.,
							vec2(5.0,1.)/8.,
							vec2(-3,-5.)/8.,
							vec2(-5.,5.)/8.,
							vec2(-7.,-1.)/8.,
							vec2(3,7.)/8.,
							vec2(7.,-7.)/8.);


#include "/lib/util.glsl"
#include "/lib/res_params.glsl"
void main() {


	tempOffsets = HaltonSeq2(frameCounter%10000);
	TAA_Offset = offsets[frameCounter%8];
	#ifndef TAA
	TAA_Offset = vec2(0.0);
	#endif

	gl_Position = ftransform();
	#ifdef TAA_UPSCALING
		gl_Position.xy = (gl_Position.xy+tempOffsets*0.5+0.5)*RENDER_SCALE*2.0-1.0;
	#endif
	#ifdef TAA
	gl_Position.xy += offsets[framemod8] * gl_Position.w*texelSize;
	#endif	
	texcoord = gl_MultiTexCoord0.xy;



	vec3 sc = texelFetch2D(colortex4,ivec2(6,37),0).rgb;
	ambientUp = texelFetch2D(colortex4,ivec2(0,37),0).rgb;
	ambientDown = texelFetch2D(colortex4,ivec2(1,37),0).rgb;
	ambientLeft = texelFetch2D(colortex4,ivec2(2,37),0).rgb;
	ambientRight = texelFetch2D(colortex4,ivec2(3,37),0).rgb;
	ambientB = texelFetch2D(colortex4,ivec2(4,37),0).rgb;
	ambientF = texelFetch2D(colortex4,ivec2(5,37),0).rgb;

	lightCol.a = float(sunElevation > 1e-5)*2-1.;
	lightCol.rgb = sc;

	WsunVec =  lightCol.a*normalize(mat3(gbufferModelViewInverse) *  sunPosition);
	zMults = vec3((far * near)*2.0,far+near,far-near);
	refractedSunVec = refract(WsunVec, -vec3(0.0,1.0,0.0), 1.0/1.33333);


}
