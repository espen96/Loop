
#extension GL_EXT_gpu_shader4 : enable
//in vec3 at_velocity;   

#extension GL_EXT_gpu_shader4 : enable


flat out vec3 zMults;
uniform float far;
uniform float near;
#include "/lib/res_params.glsl"
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
	zMults = vec3(1.0/(far * near),far+near,far-near);
	gl_Position = vec4(gl_Vertex.xy * 2.0 - 1.0, 0.0, 1.0);
	#ifdef TAA_UPSCALING
		gl_Position.xy = (gl_Position.xy*0.5+0.5)*RENDER_SCALE*2.0-1.0;
	#endif
}
