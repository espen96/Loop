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
#include "/lib/res_params.glsl"
uniform float viewWidth;
uniform float viewHeight;
out vec2 texcoord;
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
	vec2 clampedRes = max(vec2(viewWidth,viewHeight),vec2(1920.0,1080.0))/BLOOM_QUALITY;
	gl_Position = vec4(vec4(vaPosition + chunkOffset, 1.0).xy * 2.0 - 1.0, 0.0, 1.0);
	//0-0.25
	gl_Position.y = (gl_Position.y*0.5+0.5)*0.25/clampedRes.y*1080.0*2.0-1.0;
	//0-0.5
	gl_Position.x = (gl_Position.x*0.5+0.5)*0.5/clampedRes.x*1920.0*2.0-1.0;
	texcoord = vaUV0.xy/clampedRes*vec2(1920.,1080.);
}
