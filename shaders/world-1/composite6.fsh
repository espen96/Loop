#version 120
#extension GL_ARB_shader_texture_lod : enable

#include "/lib/settings.glsl"
uniform vec2 texelSize;
vec2 texcoord = gl_FragCoord.xy*texelSize;	

uniform float aspectRatio;
uniform float viewWidth;
uniform float viewHeight;

uniform sampler2D colortex3;


#ifdef FXAA
	#include "/program/comp/fxaa.glsl"
#endif

#ifndef FXAA
	void main() {
		vec3 color = texture2D(colortex3, texcoord.st).rgb;

		/* DRAWBUFFERS:3 */
		gl_FragData[0] = vec4(color, 1.0); //colortex3
	}
#endif




