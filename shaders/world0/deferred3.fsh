#version 120
#extension GL_EXT_gpu_shader4 : enable


uniform sampler2D colortex4;
uniform sampler2D depthtex1;

uniform float near;
uniform float far;


float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}
#include "/lib/res_params.glsl"
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
/* DRAWBUFFERS:4 */


//	vec3 oldTex = texelFetch2D(colortex4, ivec2(gl_FragCoord.xy), 0).xyz;
//	float newTex = linZ(texelFetch2D(depthtex1, ivec2(gl_FragCoord.xy*4), 0).x);
//	gl_FragData[0] = vec4(oldTex, newTex);
	
	
	
	vec3 oldTex = texelFetch2D(colortex4, ivec2(gl_FragCoord.xy), 0).xyz;
	float newTex = texelFetch2D(depthtex1, ivec2(gl_FragCoord.xy*4), 0).x;
  if (newTex < 1.0)
	   gl_FragData[0] = vec4(oldTex, linZ(newTex));
  else
    gl_FragData[0] = vec4(oldTex, 2.0);	
	

}
