#version 150
#extension GL_EXT_gpu_shader4 : enable
uniform vec2 texelSize;
#include "/lib/res_params.glsl"
//attribute vec3 at_velocity;   
// Compatibility
#extension GL_EXT_gpu_shader4 : enable
in vec3 vaPosition;
in vec4 vaColor;
in vec2 vaUV0;
in ivec2 vaUV2;
in vec3 vaNormal;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 textureMatrix = mat4(1.0);
uniform mat3 normalMatrix;
uniform vec3 chunkOffset;
void main() {

	gl_Position = vec4(vec4(vaPosition + chunkOffset, 1.0).xy * 2.0 - 1.0, 0.0, 1.0);
	vec2 scaleRatio = max(vec2(0.25), vec2(18.+258*2,258.)*texelSize);
	gl_Position.xy = (gl_Position.xy*0.5+0.5)*clamp(scaleRatio+0.01,0.0,1.0)*2.0-1.0;
	
	
	
}
