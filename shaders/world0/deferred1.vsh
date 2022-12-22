#version 150 compatibility
#extension GL_EXT_gpu_shader4 : enable
#define CLOUDS_QUALITY 0.35 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
// in vec3 at_velocity;

#extension GL_EXT_gpu_shader4 : enable


flat out vec3 sunColor;
flat out vec3 moonColor;
flat out vec3 avgAmbient;
flat out float tempOffsets;
out vec2 coord;

uniform sampler2D colortex4;
uniform int frameCounter;
#include "/lib/res_params.glsl"
#include "/lib/util.glsl"

void main()
{
    tempOffsets = HaltonSeq2(frameCounter % 10000);
    gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0);
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * clamp(CLOUDS_QUALITY + 0.01, 0.0, 1.0) * 2.0 - 1.0;
#ifdef TAA_UPSCALING
    gl_Position.xy = (gl_Position.xy * 0.5 + 0.5) * RENDER_SCALE * 2.0 - 1.0;
#endif
    sunColor = texelFetch2D(colortex4, ivec2(12, 37), 0).rgb;
    moonColor = texelFetch2D(colortex4, ivec2(13, 37), 0).rgb;
    avgAmbient = texelFetch2D(colortex4, ivec2(11, 37), 0).rgb;
    coord = gl_MultiTexCoord0.xy;
}
