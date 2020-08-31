#version 120
#extension GL_ARB_shader_texture_lod : enable
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
//varying float mat;
varying vec2 texcoord;
varying vec4 color;
uniform sampler2D tex;
varying vec4 lmcoord;
uniform int entityId;   
uniform int blockEntityId;
uniform sampler2D noisetex;
varying vec3 normal;
varying float iswater;
uniform float frameTimeCounter;
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
float blueNoise(){
  return texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a;
}
void main() {
/* DRAWBUFFERS:01 */




  if(blockEntityId == 1) discard;
    vec4 albedo = texture2D(tex, texcoord.xy)*color;
         albedo.rgb =  mix(albedo.rgb, mix(vec3(0.42,0.6,0.7),albedo.rgb,water_blend), iswater);
	//	 	if (albedo.a == 0.0) discard;
	//	 albedo.rgb = mix(vec3(1.0), albedo.rgb, pow(albedo.a, (1.0 - albedo.a) * 0.5) * 1.05);
	//	 albedo.rgb *= 1.0 - pow(albedo.a, 64.0);

	gl_FragData[0] = albedo;
	#ifdef SHADOW_DISABLE_ALPHA_MIPMAPS
	 gl_FragData[0].a = texture2DLod(tex,texcoord.xy,0).a;
	#endif
	#ifdef Stochastic_Transparent_Shadows
	 gl_FragData[0].a = float(gl_FragData[0].a >= blueNoise());
	 
	#endif
	gl_FragData[1] = vec4(normal.xyz * 0.5 + 0.5, gl_FragData[0].a);
}