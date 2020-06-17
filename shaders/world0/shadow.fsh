#version 120
#extension GL_ARB_shader_texture_lod : enable
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"

varying vec2 texcoord;
uniform sampler2D tex;
uniform int entityId;   
uniform int blockEntityId;
uniform sampler2D noisetex;
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
float blueNoise(){
  return texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a;
}
void main() {
  if(blockEntityId == 1) discard;

	gl_FragData[0] = texture2D(tex,texcoord.xy);
	#ifdef SHADOW_DISABLE_ALPHA_MIPMAPS
	 gl_FragData[0].a = texture2DLod(tex,texcoord.xy,0).a;
	#endif
  #ifdef Stochastic_Transparent_Shadows
	 gl_FragData[0].a = float(gl_FragData[0].a >= blueNoise());
  #endif
}
