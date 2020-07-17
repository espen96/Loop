#version 120
#extension GL_EXT_gpu_shader4 : enable

const int shadowMapResolution = 3172; //[512 768 1024 1536 2048 3172 4096 8192]

varying vec4 lmtexcoord;
varying vec4 color;



uniform sampler2D texture;
uniform sampler2D gaux1;

//faster and actually more precise than pow 2.2
vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* DRAWBUFFERS:27 */
void main() {


	gl_FragData[0] = texture2D(texture, lmtexcoord.xy);


		vec3 albedo = toLinear(gl_FragData[0].rgb*color.rgb);

		float exposure = texelFetch2D(gaux1,ivec2(10,37),0).r;

		vec3 col = albedo/exposure*0.1;


		gl_FragData[0].rgb = col*color.a;
		gl_FragData[0].a = 0.0;
	gl_FragData[1].r = 1;	


}
