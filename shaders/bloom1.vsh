#include "/lib/res_params.glsl"
uniform float viewWidth;
uniform float viewHeight;
// in vec3 at_velocity;
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
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main()
{
    // Improves performances and makes sure bloom radius stays the same at high resolution (>1080p)
    vec2 clampedRes = max(vec2(viewWidth, viewHeight), vec2(1920.0, 1080.));
    gl_Position = vec4(vec4(vaPosition + chunkOffset, 1.0).xy * 2.0 - 1.0, 0.0, 1.0);
    //*0.51 to avoid errors when sampling outside since clearing is disabled
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * 0.51 * BLOOM_QUALITY / clampedRes * vec2(1920.0, 1080.) * 2.0 - 1.0;
}
