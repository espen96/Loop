#version 120
//Reduces the resolution of the shadowmap for GI
#extension GL_EXT_gpu_shader4 : enable
varying vec2 texcoord;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowcolor1;
uniform sampler2D shadow;

uniform float viewWidth;
uniform float viewHeight;
uniform int frameCounter;
uniform float far;
uniform float near;
#include "/lib/projections.glsl"


float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}
#include "/lib/Shadow_Params.glsl"


//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
/* DRAWBUFFERS:7 */

if (gl_FragCoord.x < 512 && gl_FragCoord.y < 512 ){
vec2 tc = ((floor(gl_FragCoord.xy)+0.5)/512.*2-1.)*1.8;
float distort = calcDistort(tc);
tc=tc*distort;
tc=tc*0.5+0.5;

//gl_FragData[0]=vec4(texelFetch2D(shadowcolor0,ivec2(tc*shadowMapResolution),0).rgb,texelFetch2D(shadow,ivec2(tc*shadowMapResolution),0).x);
gl_FragData[0]=vec4(texelFetch2D(shadowcolor1,ivec2(tc*shadowMapResolution),0).rgb,1.0);
}


}
