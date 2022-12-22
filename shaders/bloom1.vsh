#include "/lib/res_params.glsl"
uniform float viewWidth;
uniform float viewHeight;
// in vec3 at_velocity;

#extension GL_EXT_gpu_shader4 : enable


//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main()
{
    // Improves performances and makes sure bloom radius stays the same at high resolution (>1080p)
    vec2 clampedRes = max(vec2(viewWidth, viewHeight), vec2(1920.0, 1080.));
    gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0);
    //*0.51 to avoid errors when sampling outside since clearing is disabled
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * 0.51 * BLOOM_QUALITY / clampedRes * vec2(1920.0, 1080.) * 2.0 - 1.0;
}
