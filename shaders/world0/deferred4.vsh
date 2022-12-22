#version 150 compatibility
#extension GL_EXT_gpu_shader4 : enable
#define TAA
// in vec3 at_velocity;

#extension GL_EXT_gpu_shader4 : enable


flat out vec3 WsunVec;
flat out vec3 ambientUp;
flat out vec3 ambientLeft;
flat out vec3 ambientRight;
flat out vec3 ambientB;
flat out vec3 ambientF;
flat out vec3 ambientDown;
flat out vec4 lightCol;
flat out float tempOffsets;
flat out vec2 TAA_Offset;
flat out vec3 zMults;
flat out vec3 refractedSunVec;

uniform sampler2D colortex4;

uniform float far;
uniform float near;
uniform mat4 gbufferModelViewInverse;
uniform vec3 sunPosition;
uniform float sunElevation;
uniform int frameCounter;

#include "/lib/kernel.glsl"

#include "/lib/res_params.glsl"
#include "/lib/util.glsl"

void main()
{
    gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0);
#ifdef TAA_UPSCALING
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * RENDER_SCALE * 2.0 - 1.0;
#endif

    tempOffsets = HaltonSeq2(frameCounter % 10000);
    TAA_Offset = offsets[frameCounter % 8];
#ifndef TAA
    TAA_Offset = vec2(0.0);
#endif

    vec3 sc = texelFetch2D(colortex4, ivec2(6, 37), 0).rgb;
    ambientUp = texelFetch2D(colortex4, ivec2(0, 37), 0).rgb;
    ambientDown = texelFetch2D(colortex4, ivec2(1, 37), 0).rgb;
    ambientLeft = texelFetch2D(colortex4, ivec2(2, 37), 0).rgb;
    ambientRight = texelFetch2D(colortex4, ivec2(3, 37), 0).rgb;
    ambientB = texelFetch2D(colortex4, ivec2(4, 37), 0).rgb;
    ambientF = texelFetch2D(colortex4, ivec2(5, 37), 0).rgb;

    lightCol.a = float(sunElevation > 1e-5) * 2 - 1.;
    lightCol.rgb = sc;

    WsunVec = lightCol.a * normalize(mat3(gbufferModelViewInverse) * sunPosition);
    zMults = vec3((far * near) * 2.0, far + near, far - near);
    refractedSunVec = refract(WsunVec, -vec3(0.0, 1.0, 0.0), 1.0 / 1.33333);
}
