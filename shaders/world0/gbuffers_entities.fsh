#version 130
#extension GL_EXT_gpu_shader4 : enable



varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;


uniform sampler2D texture;
uniform float frameTimeCounter;
uniform mat4 gbufferProjectionInverse;
uniform vec4 entityColor;
float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+frameTimeCounter*51.9521);
}

//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 n)
{

    return vec4(n.xy * inversesqrt(n.z * 8.0 + 8.0 + 0.00001) + 0.5,vec2(lmtexcoord.z,lmtexcoord.w));
}


//encoding by jodie
float encodeVec2(vec2 a){
    const vec2 constant1 = vec2( 1.0, 256.) / 65535.;
    vec2 temp = floor(a * 252.0 + 0.5);
	return temp.x*constant1.x+temp.y*constant1.y;
}
float encodeVec2(float x,float y){
    return encodeVec2(vec2(x,y));
}

													 
														
							
																													 
						  
																		  
											 
 

													 
						 
						 
						 
						 
						  
					   
						  
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
	if (data0.a > 0.3) data0.a = normalMat.a*0.5+0.1;
	else data0.a = 0.0;


	vec4 data1 = clamp(encode(normal),0.,1.0);

	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));
	gl_FragData[1] = vec4(0,2,0,0);


}
