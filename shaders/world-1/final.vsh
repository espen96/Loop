#version 120
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/res_params.glsl"	
#include "/lib/settings.glsl"

flat varying vec4 exposure;
uniform sampler2D colortex4;

//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {

	gl_Position = ftransform();
			#ifdef TAA_UPSCALING
		gl_Position.xy = (gl_Position.xy*0.5+0.5)*RENDER_SCALE*2.0-1.0;
	#endif

	exposure=vec4(texelFetch2D(colortex4,ivec2(10,37),0).r*vec3(FinalR,FinalG,FinalB),texelFetch2D(colortex4,ivec2(10,37),0).r);
}
