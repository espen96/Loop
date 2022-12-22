
#extension GL_EXT_gpu_shader4 : enable

//in vec3 at_velocity;   

#extension GL_EXT_gpu_shader4 : enable


out vec2 texcoord;
flat out float exposureA;
flat out float tempOffsets;
uniform sampler2D colortex4;
uniform int frameCounter;
#include "/lib/util.glsl"
void main() {

	tempOffsets = HaltonSeq2(frameCounter%10000);
	gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0);
	texcoord = gl_MultiTexCoord0.xy;
	exposureA = texelFetch2D(colortex4,ivec2(10,37),0).r;
}
