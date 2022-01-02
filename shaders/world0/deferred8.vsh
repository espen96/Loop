#version 150
#extension GL_EXT_gpu_shader4 : enable
#define TAA
#define FinalR                                                                                                         \
    1.0 //[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9
        //2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9
        //4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9
        //6 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9
        //8 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0]
#define FinalG                                                                                                         \
    1.0 //[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9
        //2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9
        //4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9
        //6 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9
        //8 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0]
#define FinalB                                                                                                         \
    1.0 //[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9
        //2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9
        //4 4.1 4.2 4.3 4.4 4.5 4.6 4.7 4.8 4.9 5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9
        //6 6.1 6.2 6.3 6.4 6.5 6.6 6.7 6.8 6.9 7 7.1 7.2 7.3 7.4 7.5 7.6 7.7 7.8 7.9
        //8 8.1 8.2 8.3 8.4 8.5 8.6 8.7 8.8 8.9 9 9.1 9.2 9.3 9.4 9.5 9.6 9.7 9.8 9.9 10.0]
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
flat out vec4 exposure;
uniform sampler2D colortex4;

flat out vec2 coord;

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
    zMults = vec3(1.0 / (far * near), far + near, far - near);
    gl_Position = vec4(vec4(vaPosition + chunkOffset, 1.0).xy * 2.0 - 1.0, 0.0, 1.0);
    //	gl_Position.xy = (gl_Position.xy*0.5+0.5)*(0.01+VL_RENDER_RESOLUTION)*2.0-1.0;
#ifdef TAA_UPSCALING
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * RENDER_SCALE * 2.0 - 1.0;
#endif

    tempOffsets = HaltonSeq2(frameCounter % 10000);
    TAA_Offset = offsets[frameCounter % 8];
#ifndef TAA
    TAA_Offset = vec2(0.0);
#endif
    coord = vaUV0.xy;
    vec3 sc = texelFetch2D(colortex4, ivec2(6, 37), 0).rgb;
    ambientUp = texelFetch2D(colortex4, ivec2(0, 37), 0).rgb;
    ambientDown = texelFetch2D(colortex4, ivec2(1, 37), 0).rgb;
    ambientLeft = texelFetch2D(colortex4, ivec2(2, 37), 0).rgb;
    ambientRight = texelFetch2D(colortex4, ivec2(3, 37), 0).rgb;
    ambientB = texelFetch2D(colortex4, ivec2(4, 37), 0).rgb;
    ambientF = texelFetch2D(colortex4, ivec2(5, 37), 0).rgb;
    exposure = vec4(texelFetch2D(colortex4, ivec2(10, 37), 0).r * vec3(FinalR, FinalG, FinalB),
                    texelFetch2D(colortex4, ivec2(10, 37), 0).r);
    lightCol.a = float(sunElevation > 1e-5) * 2 - 1.;
    lightCol.rgb = sc;

    WsunVec = lightCol.a * normalize(mat3(gbufferModelViewInverse) * sunPosition);
    zMults = vec3((far * near) * 2.0, far + near, far - near);
    refractedSunVec = refract(WsunVec, -vec3(0.0, 1.0, 0.0), 1.0 / 1.33333);
}
