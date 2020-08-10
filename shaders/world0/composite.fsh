#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D depthtex1;
uniform sampler2D colortex3;
uniform vec2 texelSize;



#include "/lib/settings.glsl"


flat varying float tempOffsets;
flat varying vec3 WsunVec;
flat varying vec2 TAA_Offset;

#include "/lib/res_params.glsl"							   
#include "/lib/Shadow_Params.glsl"

flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;							  
uniform sampler2D shadow;
uniform sampler2D colortex1;

uniform sampler2D noisetex;
uniform vec3 sunVec;
uniform float far;
uniform float near;

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
  return texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a;
}
vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
	// l = (2*n)/(f+n-d(f-n))
	// f+n-d(f-n) = 2n/l
	// -d(f-n) = ((2n/l)-f-n)
	// d = -((2n/l)-f-n)/(f-n)

}
float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
}




void main() {
/* DRAWBUFFERS:3 */
	vec2 texcoord = ((gl_FragCoord.xy))*texelSize;

	gl_FragData[0] = vec4(Min_Shadow_Filter_Radius, 0.0, 0.0, 0.0);
	float z = texture2D(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;	
	float masks = texture2D(colortex3,texcoord).a;
	gl_FragData[0].a = masks;
	if (z < 1.0){

		vec4 data = texture2D(colortex1,texcoord);
	
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);
		vec2 lightmap = dataUnpacked1.yz;
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		vec2 spec = texture2D(colortex3,texcoord).gb;

		if (!hand){
			gl_FragData[0].gb =spec;		
		
		
			float NdotL = clamp(dot(normal,WsunVec),0.0,1.0);
			if (translucent) {
				NdotL = 0.9;
			}

				float bn = blueNoise(gl_FragCoord.xy);
				float noise = fract(bn + frameCounter/1.6180339887);
				vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
				#ifdef Variable_Penumbra_Shadows
					if (NdotL > 0.001) {
					vec3 p3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;
					vec3 projectedShadowPosition = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
					projectedShadowPosition = diagonal3(shadowProjection) * projectedShadowPosition + shadowProjection[3].xyz;

					//apply distortion
					float distortFactor = calcDistort(projectedShadowPosition.xy);
					projectedShadowPosition.xy *= distortFactor;
					//do shadows only if on shadow map
					if (abs(projectedShadowPosition.x) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.y) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.z) < 6.0){
						const float threshMul = max(2048.0/shadowMapResolution*shadowDistance/128.0,0.95);
						float distortThresh = (sqrt(1.0-NdotL*NdotL)/NdotL+0.7)/distortFactor;
						float diffthresh =  translucent? 0.00014/15.0 : distortThresh/7500.0*threshMul;
						projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);


						const float mult = Max_Shadow_Filter_Radius;
						float avgBlockerDepth = 0.0;
						vec2 scales = vec2(0.0,Max_Filter_Depth);
						float blockerCount = 0.0;
						float rdMul = distortFactor*(1.0+mult)*d0*k/shadowMapResolution;
						float diffthreshM = diffthresh*mult*distortFactor*d0*k;
						for(int i = 0; i < VPS_Search_Samples; i++){
							vec2 offsetS = tapLocation(i,VPS_Search_Samples, 67.0,noise,0.0);

							float d = texelFetch2D( shadow, ivec2((projectedShadowPosition.xy+offsetS*rdMul)*shadowMapResolution),0).x;
							float b  = ffstep(d,projectedShadowPosition.z-i*diffthreshM/VPS_Search_Samples-diffthreshM);

							blockerCount += b;
							avgBlockerDepth += d * b;
						}
						if (blockerCount >= 0.9){
							avgBlockerDepth /= blockerCount;
							float ssample = max(projectedShadowPosition.z - avgBlockerDepth,0.0)*1500.0;
							gl_FragData[0].r = clamp(ssample, scales.x, scales.y)/(scales.y)*(mult-Min_Shadow_Filter_Radius)+Min_Shadow_Filter_Radius;

						}
					}
				}
			#endif
		}

}



}
