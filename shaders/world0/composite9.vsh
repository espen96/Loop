#version 120
#extension GL_EXT_gpu_shader4 : enable

flat varying vec2 rodExposureDepth;
varying vec2 texcoord;
flat varying float exposureA;
flat varying float tempOffsets;
uniform sampler2D colortex4;
uniform int frameCounter;
#include "/lib/util.glsl"
void main() {

	tempOffsets = HaltonSeq2(frameCounter%10000);
	gl_Position = ftransform();
	texcoord = gl_MultiTexCoord0.xy;
	exposureA = texelFetch2D(colortex4,ivec2(10,37),0).r;	
	rodExposureDepth = texelFetch2D(colortex4,ivec2(14,37),0).rg;
	rodExposureDepth.y = sqrt(rodExposureDepth.y/65000.0);
}
