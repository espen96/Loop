#version 120
#include "/lib/res_params.glsl"

/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/

varying vec4 lmtexcoord;
varying vec4 color;

uniform vec2 texelSize;
uniform int framemod8;
uniform vec3 cameraPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform float frameTimeCounter;
const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
												vec2(-1.,3.)/8.,
												vec2(5.0,1.)/8.,
												vec2(-3,-5.)/8.,
												vec2(-5.,5.)/8.,
												vec2(-7.,-1.)/8.,
												vec2(3,7.)/8.,
												vec2(7.,-7.)/8.);
const float PI48 = 150.796447372;
float pi2wt = PI48*frameTimeCounter*10.;

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec4 toClipSpace3(vec3 viewSpacePosition) {
    return vec4(projMAD(gl_ProjectionMatrix, viewSpacePosition),-viewSpacePosition.z);
}
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {

	lmtexcoord.xy = (gl_MultiTexCoord0).xy;

	vec2 lmcoord = gl_MultiTexCoord1.xy/255.;
	lmtexcoord.zw = lmcoord*lmcoord;


	vec3 position = mat3(gl_ModelViewMatrix) * vec3(gl_Vertex) + gl_ModelViewMatrix[3].xyz;
	vec3 worldpos = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz + cameraPosition;
	bool istopv = worldpos.y > cameraPosition.y+5.0;
	float ft = frameTimeCounter*1.3;
	if (!istopv) position.xz += vec2(3.0,1.0)+sin(ft)*sin(ft)*sin(ft)*vec2(2.1,0.6);
	position.xz -= (vec2(3.0,1.0)+sin(ft)*sin(ft)*sin(ft)*vec2(2.1,0.6))*0.5;
	gl_Position = toClipSpace3(position);

	color = gl_Color;
	#ifdef TAA_UPSCALING
		gl_Position.xy = gl_Position.xy * RENDER_SCALE + RENDER_SCALE * gl_Position.w - gl_Position.w;
	#endif
	#ifdef TAA
	gl_Position.xy += offsets[framemod8] * gl_Position.w*texelSize;
	#endif
}
