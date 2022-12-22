#version 150 compatibility
#extension GL_EXT_gpu_shader4 : enable
#define TAA
flat out vec2 TAA_Offset;
flat out vec3 WsunVec;
out vec2 coord;
uniform sampler2D colortex4;
// in vec3 at_velocity;

#extension GL_EXT_gpu_shader4 : enable


uniform int frameCounter;
uniform float sunElevation;
uniform vec3 sunPosition;
uniform mat4 gbufferModelViewInverse;
#include "/lib/res_params.glsl"
#include "/lib/util.glsl"

const vec2[8] offsets = vec2[8](vec2(1. / 8., -3. / 8.), vec2(-1., 3.) / 8., vec2(5.0, 1.) / 8., vec2(-3, -5.) / 8.,
                                vec2(-5., 5.) / 8., vec2(-7., -1.) / 8., vec2(3, 7.) / 8., vec2(7., -7.) / 8.);
void main()
{
    TAA_Offset = offsets[frameCounter % 8];
#ifndef TAA
    TAA_Offset = vec2(0.0);
#endif
    gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0);
#ifdef TAA_UPSCALING
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * RENDER_SCALE * 2.0 - 1.0;
#endif
    coord = gl_MultiTexCoord0.xy;
    WsunVec = (float(sunElevation > 1e-5) * 2 - 1.) * normalize(mat3(gbufferModelViewInverse) * sunPosition);
}
