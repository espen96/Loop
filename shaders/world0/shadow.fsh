#version 150
// moved up
#extension GL_ARB_shader_texture_lod : enable
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/res_params.glsl"

//#define SHADOW_DISABLE_ALPHA_MIPMAPS // Disables mipmaps on the transparency of alpha-tested things like foliage, may cost a few fps in some cases
//#define Stochastic_Transparent_Shadows // Highly recommanded to enable SHADOW_DISABLE_ALPHA_MIPMAPS with it. Uses noise to simulate transparent objects' shadows (not colored). It is also recommended to increase Min_Shadow_Filter_Radius with this.
in vec2 texcoord;
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
	vec4 albedo = texture(tex,texcoord.xy);	
	#ifdef SHADOW_DISABLE_ALPHA_MIPMAPS
	 albedo.a = textureLod(tex,texcoord.xy,0).a;
	#endif

//	if (albedo.a < 0.01) discard;


//	albedo.rgb = mix(vec3(1.0), albedo.rgb, 1.0 - pow(1.0 - albedo.a, 3.0));
//	albedo.rgb *= clamp(1.0 - pow(albedo.a, 64.0),0,1);




	gl_FragData[0] = albedo;

	

	
  #ifdef Stochastic_Transparent_Shadows
	 gl_FragData[0].a = float(gl_FragData[0].a >= blueNoise());
  #endif
}
