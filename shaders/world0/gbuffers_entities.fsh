#version 120
#extension GL_EXT_gpu_shader4 : enable


#define gbuff_ent

#include "/program/gbuffers/standard.glsl"





#include "/lib/encode.glsl"
	   
						  
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* DRAWBUFFERS:17 */
void main() {
	float noise = interleaved_gradientNoise();
	vec3 normal = normalMat.xyz;

	vec4 data0 = texture2D(texture, lmtexcoord.xy)*color;
	data0.rgb = mix(data0.rgb,entityColor.rgb,entityColor.a);
	if (data0.a > 0.2475) data0.a = normalMat.a*0.5+0.1;
	else data0.a = 0.0;


	vec4 data1 = clamp(encode(normal),0.,1.0);

	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encode2Vec2(data1.w,data0.w));
	gl_FragData[1] = vec4(0,2,0,0);


}
