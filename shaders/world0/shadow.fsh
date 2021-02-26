#version 120
#include "/lib/res_params.glsl"
#extension GL_ARB_shader_texture_lod : enable
#extension GL_EXT_gpu_shader4 : enable
//#define SHADOW_DISABLE_ALPHA_MIPMAPS // Disables mipmaps on the transparency of alpha-tested things like foliage, may cost a few fps in some cases
//#define Stochastic_Transparent_Shadows // Highly recommanded to enable SHADOW_DISABLE_ALPHA_MIPMAPS with it. Uses noise to simulate transparent objects' shadows (not colored). It is also recommended to increase Min_Shadow_Filter_Radius with this.
varying vec2 texcoord;
uniform sampler2D tex;
uniform sampler2D noisetex;
uniform int blockEntityId;
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
float blueNoise(){
  return texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a;
}
void main() {
	if (blockEntityId == 80) discard;
	vec4 albedo = texture2D(tex,texcoord.xy);	
	#ifdef SHADOW_DISABLE_ALPHA_MIPMAPS
	 albedo.a = texture2DLod(tex,texcoord.xy,0).a;
	#endif
	gl_FragData[0] = albedo;
	if ( albedo.a <= 0.0) discard;	
	

	
  #ifdef Stochastic_Transparent_Shadows
	 gl_FragData[0].a = float(gl_FragData[0].a >= blueNoise());
  #endif
}
