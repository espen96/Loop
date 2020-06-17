#version 120
#extension GL_EXT_gpu_shader4 : enable
#extension GL_ARB_shader_texture_lod : enable




varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;


uniform sampler2D texture;
uniform float frameTimeCounter;
uniform mat4 gbufferProjectionInverse;
float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+frameTimeCounter*51.9521);
}

//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 n)
{

    return vec4(n.xy*inversesqrt(n.z*8.0+8.0) + 0.5,vec2(lmtexcoord.z,lmtexcoord.w));
}


//encoding by jodie
float encodeVec2(vec2 a){
    const vec2 constant1 = vec2( 1., 256.) / 65535.;
    vec2 temp = floor( a * 255. );
	return temp.x*constant1.x+temp.y*constant1.y;
}
float encodeVec2(float x,float y){
    return encodeVec2(vec2(x,y));
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}

float luma(vec3 color) {
	return sqrt(dot(color*color,vec3(0.299, 0.587, 0.114)));
}

		const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
									vec2(-1.,3.)/8.,
									vec2(5.0,1.)/8.,
									vec2(-3,-5.)/8.,
									vec2(-5.,5.)/8.,
									vec2(-7.,-1.)/8.,
									vec2(3,7.)/8.,
									vec2(7.,-7.)/8.);
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* DRAWBUFFERS:1 */
void main() {
	float noise = interleaved_gradientNoise();
	vec3 normal = normalMat.xyz;

	vec4 data0 = texture2D(texture, lmtexcoord.xy);
  #ifdef DISABLE_ALPHA_MIPMAPS
  data0.a = texture2DLod(texture,lmtexcoord.xy,0).a;
  #endif

	data0.rgb*=color.rgb;
	if (data0.a > 0.1) data0.a = normalMat.a*0.5+0.5;



	vec4 data1 = clamp(noise/256.+encode(normal),0.,1.0);

	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));
}
