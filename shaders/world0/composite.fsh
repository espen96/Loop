#version 120
#extension GL_EXT_gpu_shader4 : enable



#include "/lib/settings.glsl"
flat varying float tempOffsets;
flat varying vec3 WsunVec;
flat varying vec2 TAA_Offset;
#include "/lib/Shadow_Params.glsl"
uniform sampler2D depthtex1;
uniform sampler2D shadow;
uniform sampler2D colortex1;
uniform sampler2D colortex7;
uniform sampler2D noisetex;
uniform vec3 sunVec;
uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform int frameCounter;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform vec3 cameraPosition;
uniform float viewWidth;
uniform float aspectRatio;
uniform float viewHeight;
#include "/lib/encode.glsl"



#define ffstep(x,y) clamp((y - x) * 1e35,0.0,1.0)
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter,float distort)
{
    float alpha = (sampleNumber+jitter)/nb;
    float angle = jitter*6.28+alpha * nbRot * 6.28;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*alpha;
}
float interleaved_gradientNoise(){
	vec2 coord = gl_FragCoord.xy;
	float noise = fract(52.9829189*fract(0.06711056*coord.x + 0.00583715*coord.y));
	return noise;
}


float hash13(vec3 p3)
{
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}
float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y);
}
float blueNoise(vec2 coord){
  return texelFetch2D(noisetex, ivec2(coord)%512, 0).a;
}
vec2 tapLocation(int sampleNumber, float spinAngle,int nb, float nbRot,float r0)
{
    float alpha = (float(sampleNumber*1.0f + r0) * (1.0 / (nb)));
    float angle = alpha * (nbRot * 6.28) + spinAngle*6.28;

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}

void ssao(inout float occlusion,vec3 fragpos,float mulfov,float dither,vec3 normal)
{

	ivec2 pos = ivec2(gl_FragCoord.xy);
	const float tan70 = tan(70.*3.14/180.);
	float mulfov2 = gbufferProjection[1][1]/tan70;


	float maxR2 = fragpos.z*fragpos.z*mulfov2*2.*1.412/50.0;



	float rd = mulfov2*0.04;
	//pre-rotate direction
	float n = 0.;

	occlusion = 0.0;

	vec2 acc = -vec2(TAA_Offset)*texelSize*0.5;
	float mult = (dot(normal,normalize(fragpos))+1.0)*0.5+0.5;

	vec2 v = fract(vec2(dither,interleaved_gradientNoise()) + (frameCounter%10000) * vec2(0.75487765, 0.56984026));
	for (int j = 0; j < SSAO_SAMPLES ;j++) {

			vec2 sp = tapLocation(j,v.x,SSAO_SAMPLES,5.,v.y);
			vec2 sampleOffset = sp*rd;
			ivec2 offset = ivec2(gl_FragCoord.xy + sampleOffset*vec2(viewWidth,viewHeight*aspectRatio));
			if (offset.x >= 0 && offset.y >= 0 && offset.x < viewWidth && offset.y < viewHeight ) {
				vec3 t0 = toScreenSpace(vec3(offset*texelSize+acc+0.5*texelSize,texelFetch2D(depthtex1,offset,0).x));

				vec3 vec = t0.xyz - fragpos;
				float dsquared = dot(vec,vec);
				if (dsquared > 1e-5){
					if (dsquared < maxR2){
						float NdotV = clamp(dot(vec*inversesqrt(dsquared), normalize(normal)),0.,1.);
						occlusion += NdotV * clamp(1.0-dsquared/maxR2,0.0,1.0);
					}
					n += 1.0;
				}
			}
		}



		occlusion = clamp(1.0-occlusion/n*SSAO_STRENGTH,0.,1.0);
		//occlusion = mult;

}


void main() {
/* DRAWBUFFERS:3 */


	vec2 texcoord = ((gl_FragCoord.xy))*texelSize;
	gl_FragData[0] = vec4(Min_Shadow_Filter_Radius,1.0,0.0,0.0);
	float z = texture2D(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	if (z < 1.0){

		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);

		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		if (!hand){
			float NdotL = clamp(dot(normal,WsunVec),0.0,1.0);
			if (translucent) {
				NdotL = 0.9;
			}

				float bn = blueNoise(gl_FragCoord.xy);
				float noise = fract(bn + frameCounter/1.6180339887);
				vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(tempOffset)*texelSize*0.5,z));
			float ao= 1.0;
			#ifdef SSAO
				ssao(ao,fragpos,1.0,noise,decode(dataUnpacked0.yw));
				gl_FragData[0].g = ao;
			#endif
		}

}
}
