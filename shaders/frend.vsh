
#extension GL_EXT_gpu_shader4 : enable
//in vec3 at_velocity;   
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
flat out vec3 zMults;
uniform float far;
uniform float near;
#include "/lib/res_params.glsl"
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
	zMults = vec3(1.0/(far * near),far+near,far-near);
	gl_Position = vec4(vec4(vaPosition + chunkOffset, 1.0).xy * 2.0 - 1.0, 0.0, 1.0);
	#ifdef TAA_UPSCALING
		gl_Position.xy = (gl_Position.xy*0.5+0.5)*RENDER_SCALE*2.0-1.0;
	#endif
}
