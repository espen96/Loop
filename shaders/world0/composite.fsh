#version 130
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
//#define POM


#define SSAO

#define ROUGHREF
#define power



#define Depth_Write_POM	// POM adjusts the actual position, so screen space shadows can cast shadows on POM
#define POM_DEPTH 0.25 // [0.025 0.05 0.075 0.1 0.125 0.15 0.20 0.25 0.30 0.50 0.75 1.0] //Increase to increase POM strength
#define CAVE_LIGHT_LEAK_FIX // Hackish way to remove sunlight incorrectly leaking into the caves. Can inacurrately create shadows in some places
//#define CLOUDS_SHADOWS
#define CLOUDS_SHADOWS_STRENGTH 1.0 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define SPECSTRENGTH 1.0 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define SSPTMIX1 1.0 //[0.0 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define SSPTambient 1.0 // [0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.20 0.22 0.23 0.25 0.27 0.29 0.32 0.34 0.37 0.40 0.43 0.46 0.50 0.54 0.58 0.63 0.68 0.74 0.79 0.86 0.93 1.00 1.08 1.17 1.26 1.36 1.47 1.59 1.71 1.85 2.00 2.16 2.33 2.51 2.72 2.93 3.17 3.42 3.69 3.99 4.30 4.65 5.02 5.42 5.85 6.32 6.82 7.37 7.95 8.59 9.27 10.01 10.81 11.68 12.61 13.61 14.70 15.87 17.14 18.51 19.99 21.58 23.30 25.16 27.17 29.34 31.68 34.21 36.94 39.89 43.07 46.50 50.22 54.22 58.55 63.22 68.27 73.72 79.60 85.95 92.81 100.22 108.21 116.85 126.17 136.24 147.11 158.85 171.53 185.22 200.00]

#define CLOUDS_QUALITY 0.35 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define SPEC_SSR_QUALITY 3 //[1 2 3 4 5 6 7 8 9 10 ]


#define TORCH_R 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_G 0.4 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_B 0.13 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]

#define Emissive_Strength 1.00 // [0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.20 0.22 0.23 0.25 0.27 0.29 0.32 0.34 0.37 0.40 0.43 0.46 0.50 0.54 0.58 0.63 0.68 0.74 0.79 0.86 0.93 1.00 1.08 1.17 1.26 1.36 1.47 1.59 1.71 1.85 2.00 2.16 2.33 2.51 2.72 2.93 3.17 3.42 3.69 3.99 4.30 4.65 5.02 5.42 5.85 6.32 6.82 7.37 7.95 8.59 9.27 10.01 10.81 11.68 12.61 13.61 14.70 15.87 17.14 18.51 19.99 21.58 23.30 25.16 27.17 29.34 31.68 34.21 36.94 39.89 43.07 46.50 50.22 54.22 58.55 63.22 68.27 73.72 79.60 85.95 92.81 100.22 108.21 116.85 126.17 136.24 147.11 158.85 171.53 185.22 200.00]


const bool shadowHardwareFiltering = true;

flat varying vec4 lightCol; //main light source color (rgb),used light source(1=sun,-1=moon)
flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;
flat varying vec3 WsunVec;
flat varying vec2 TAA_Offset;
flat varying float tempOffsets;
flat varying vec3 refractedSunVec;




uniform sampler2D colortex0;//clouds
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortex3;
uniform sampler2D colortex2;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;
uniform sampler2D colortexA;
uniform sampler2D colortexB;


uniform sampler2D colortexC;
uniform sampler2D colortexE;


uniform sampler2D colortex6; // Noise
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth

uniform sampler2D shadow;
uniform sampler2D shadowcolor1;
uniform sampler2D shadowcolor0;
uniform mat4 shadowProjectionInverse;

uniform int framemod8;
uniform int heldBlockLightValue;
uniform int frameCounter;
uniform int isEyeInWater;
uniform float far;
uniform float wetness;
uniform float near;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 previousCameraPosition;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 gbufferModelView;
uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;
uniform vec2 texelSize;
uniform vec3 cameraPosition;
uniform vec3 sunVec;
uniform ivec2 eyeBrightnessSmooth;

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
vec3 toScreenSpacePrev(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}

#include "/lib/res_params.glsl"
#include "/lib/waterOptions.glsl"
#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/stars.glsl"
#include "/lib/volumetricClouds.glsl"

float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}

uniform sampler2D colortexD;
#include "/lib/specular.glsl"
vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}
float lengthVec (vec3 vec){
	return sqrt(dot(vec,vec));
}
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
float triangularize(float dither)
{
    float center = dither*2.0-1.0;
    dither = center*inversesqrt(abs(center));
    return clamp(dither-fsign(center),0.0,1.0);
}
float interleaved_gradientNoise(float temp){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+temp);
}
vec3 fp10Dither(vec3 color,float dither){
	const vec3 mantissaBits = vec3(6.,6.,5.);
	vec3 exponent = floor(log2(color));
	return color + dither*exp2(-mantissaBits)*exp2(exponent);
}



float facos(float sx){
    float x = clamp(abs( sx ),0.,1.);
    return sqrt( 1. - x ) * ( -0.16882 * x + 1.56734 );
}
vec3 decode (vec2 encn)
{
    vec3 unenc = vec3(0.0);
    encn = encn * 2.0 - 1.0;
    unenc.xy = abs(encn);
    unenc.z = 1.0 - unenc.x - unenc.y;
    unenc.xy = unenc.z <= 0.0 ? (1.0 - unenc.yx) * sign(encn) : encn;
    return normalize(unenc.xyz);
}

vec2 decodeVec2(float a){
    const vec2 constant1 = 65535. / vec2( 256., 65536.);
    const float constant2 = 256. / 255.;
    return fract( a * constant1 ) * constant2 ;
}
float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
	// l = (2*n)/(f+n-d(f-n))
	// f+n-d(f-n) = 2n/l
	// -d(f-n) = ((2n/l)-f-n)
	// d = -((2n/l)-f-n)/(f-n)

}







vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter,float distort)
{
		float alpha0 = sampleNumber/nb;
    float alpha = (sampleNumber+jitter)/nb;
    float angle = jitter*6.28 + alpha * 84.0 * 6.28;

    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*sqrt(alpha);
}



vec3 BilateralFiltering(sampler2D tex, sampler2D depth,vec2 coord,float frDepth,float maxZ){
  vec4 sampled = vec4(texelFetch2D(tex,ivec2(coord),0).rgb,1.0);

  return vec3(sampled.x,sampled.yz/sampled.w);
}
float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}
vec4 blueNoise(vec2 coord){
  return texelFetch2D(colortex6, ivec2(coord)%512, 0);
}
float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * frameCounter);
}
vec3 toShadowSpaceProjected(vec3 p3){
    p3 = mat3(gbufferModelViewInverse) * p3 + gbufferModelViewInverse[3].xyz;
    p3 = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
    p3 = diagonal3(shadowProjection) * p3 + shadowProjection[3].xyz;

    return p3;
}

vec2 tapLocation(int sampleNumber, float spinAngle,int nb, float nbRot,float r0)
{
    float alpha = (float(sampleNumber + r0) * (1.0 / (nb)));
    float angle = alpha * (nbRot * 6.28) + spinAngle*6.28;

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}

#include "/lib/rsm.glsl"
#include "/lib/filter.glsl"
#include "/lib/ssgi.glsl"




void ssao(inout float occlusion,vec3 fragpos,float mulfov,float dither,vec3 normal, float z)
{

	const float tan70 = tan(70.*3.14/180.);
	float mulfov2 = gbufferProjection[1][1]/tan70;


	float maxR2 = fragpos.z*fragpos.z*mulfov2*2.*1.412/50.0;



	float rd = mulfov2*0.04;	//pre-rotate direction
	
	float n = 0.;

	occlusion = 0.0;
	
	int samples = 6;


	vec2 acc = -vec2(TAA_Offset)*texelSize*0.5;
	float mult = (dot(normal,normalize(fragpos))+1.0)*0.5+0.5;

	vec2 v = fract(vec2(dither,blueNoise()) + (frameCounter%10000) * vec2(0.75487765, 0.56984026));
	for (int j = 0; j < samples ;j++) {

			vec2 sp = tapLocation(j,v.x,7,88.,v.y);
			vec2 sampleOffset = sp*rd;
			ivec2 offset = ivec2(gl_FragCoord.xy + sampleOffset*vec2(viewWidth,viewHeight*aspectRatio)*RENDER_SCALE);
			if (offset.x >= 0 && offset.y >= 0 && offset.x < viewWidth*RENDER_SCALE.x && offset.y < viewHeight*RENDER_SCALE.y ) {
				vec3 t0 = toScreenSpace(vec3(offset*texelSize+acc+0.5*texelSize,texelFetch2D(depthtex1,offset,0).x) * vec3(1.0/RENDER_SCALE, 1.0));

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



		occlusion = clamp(1.0-occlusion/n*1.6,0.,1.0);
		//occlusion = mult;


}
vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos;

    return pos.xyz;
}

vec3 viewToWorld(vec3 viewPos) {

    vec4 pos;
    pos.xyz = viewPos;
    pos.w = 0.0;
    pos = gbufferModelViewInverse * pos;

    return pos.xyz;
}

float encodeVec2v2(vec2 a){
    ivec2 bf = ivec2(a*255.);
    return float( bf.x|(bf.y<<8) ) / 65535.;
}


vec3 decode3x16(float a){
    int bf = int(a*65535.);
    return vec3(bf%32, (bf>>5)%64, bf>>11) / vec3(31,63,31);
}

float encode2x16(vec2 a){
    ivec2 bf = ivec2(a*255.);
    return float( bf.x|(bf.y<<8) ) / 65535.;
}

vec2 decode2x16(float a){
    int bf = int(a*65535.);
    return vec2(bf%256, bf>>8) / 255.;
}
float encodeNormal3x16(vec3 a){
    vec3 b  = abs(a);
    vec2 p  = a.xy / (b.x + b.y + b.z);
    vec2 sp = vec2(greaterThanEqual(p, vec2(0.0))) * 2.0 - 1.0;

    vec2 encoded = a.z <= 0.0 ? (1.0 - abs(p.yx)) * sp : p;

    encoded = encoded * 0.5 + 0.5;

    return encode2x16(encoded);
}

vec3 decodeNormal3x16(float encoded){
    vec2 a = decode2x16(encoded);

    a = a * 2.0 - 1.0;
    vec2 b = abs(a);
    float z = 1.0 - b.x - b.y;
    vec2 sa = vec2(greaterThanEqual(a, vec2(0.0))) * 2.0 - 1.0;

    vec3 decoded = normalize(vec3(
        z < 0.0 ? (1.0 - b.yx) * sa : a.xy,
        z
    ));

    return decoded;
}
void main() {
	vec2 texcoord = gl_FragCoord.xy*texelSize;

	float z0 = texture2D(depthtex0,texcoord).x;
	float z = texture2D(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();


	vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normalize(p3);
		vec3 directLightCol = lightCol.rgb;

	if (z <=1.0) {

		p3 += gbufferModelViewInverse[3].xyz;

		float edgemask = clamp(edgefilter(texcoord*RENDER_SCALE,2,colortex8).rgb,0,1).r;

		vec4 trpData = texture2D(colortex7,texcoord);
		bool iswater = texture2D(colortex7,texcoord).a > 0.99;
		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		vec4 transparent = texture2D(colortex2,texcoord);
		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
//		vec3 normal = mat3(gbufferModelViewInverse) * worldToView(decode(dataUnpacked0.yw));
		vec3 normal = mat3(gbufferModelViewInverse) * worldToView(decode(dataUnpacked0.yw));
		vec3 normal2 =  worldToView(decode(dataUnpacked0.yw));
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;


		vec2 lightmap = dataUnpacked1.yz;
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;	// Strong translucency
		bool translucent2 = abs(dataUnpacked1.w-0.6) <0.01;	// Weak translucency
	
		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		
		float NdotLGeom = dot(normal, WsunVec);
		float NdotL = NdotLGeom;
		if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater == 1))
			NdotL = dot(normal, refractedSunVec);

		float diffuseSun = clamp(NdotL,0.,1.0);
		vec3 filtered = vec3(1.412,1.0,0.0);
		if (!hand){
			filtered = texture2D(colortex3,texcoord).rgb;
		}
		float shading = 1.0 - filtered.b;
		
					vec3 shadowCol = vec3(0.0);
				//	shadowCol = ((getRSM(normal,false,albedo, lightmap,z)) * 5)*lightmap.y;
				//	float lum = luma(shadowCol);
				//	vec3 diff = shadowCol-lum;		


  
				//	#define GISAT 10.0
				//	#define GICROSS -10.0
				
				//	shadowCol = clamp(shadowCol + diff*(-lum*(GICROSS) + GISAT),0,1);

		


		vec3 SSS = vec3(0.0);
		float sssAmount = 0.0;

		#ifdef Variable_Penumbra_Shadows
		// compute shadows only if not backfacing the sun
		// or if the blocker search was full or empty
		// always compute all shadows at close range where artifacts may be more visible
		if (diffuseSun > 0.001) {
		#else
		if (translucent) {
			sssAmount = 0.5;
			diffuseSun = mix(max(phaseg(dot(np3, WsunVec),0.5), 2.0*phaseg(dot(np3, WsunVec),0.1))*3.14150*1.6, diffuseSun, 0.3);
		}
		if (diffuseSun > 0.000) {
		#endif
			vec3 projectedShadowPosition = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
			projectedShadowPosition = diagonal3(shadowProjection) * projectedShadowPosition + shadowProjection[3].xyz;
			//apply distortion
			float distortFactor = calcDistort(projectedShadowPosition.xy);
			projectedShadowPosition.xy *= distortFactor;
			//do shadows only if on shadow map
			if (abs(projectedShadowPosition.x) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.y) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.z) < 6.0){
				float rdMul = filtered.x*distortFactor*d0*k/shadowMapResolution;
				const float threshMul = max(2048.0/shadowMapResolution*shadowDistance/128.0,0.95);
				float distortThresh = (sqrt(1.0-NdotLGeom*NdotLGeom)/NdotLGeom+0.7)/distortFactor;
				#ifdef Variable_Penumbra_Shadows
				float diffthresh = distortThresh/6000.0*threshMul;
				#else
				float diffthresh = translucent? 0.0001 : distortThresh/6000.0*threshMul;
				#endif
				#ifdef POM
				#ifdef Depth_Write_POM
					diffthresh += POM_DEPTH*0.0078125*0.25*0.166;
				#endif
				#endif
				projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);
				shading = 0.0;
				for(int i = 0; i < SHADOW_FILTER_SAMPLE_COUNT; i++){
					vec2 offsetS = tapLocation(i,SHADOW_FILTER_SAMPLE_COUNT, 0.0,noise,0.0);

					float weight = 1.0+(i+noise)*rdMul/SHADOW_FILTER_SAMPLE_COUNT*shadowMapResolution;
			//		float isShadow = shadow2D(shadow,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x;
					float isShadow = texture2D(shadow,vec2(projectedShadowPosition.xy)).x;
					shading += isShadow/SHADOW_FILTER_SAMPLE_COUNT;
				}
			}
		}

		//custom shading model for translucent objects
		#ifdef Variable_Penumbra_Shadows
		if (translucent) {
			sssAmount = 0.5;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.5*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			diffuseSun *= 1.0 - sssAmount;
			SSS *= sqrt(lightmap.y);
		}

		if (translucent2) {
			sssAmount = 0.2;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			diffuseSun *= 1.0 - sssAmount;
			SSS *= sqrt(lightmap.y);
		}
		#endif


		if ((diffuseSun*shading > 0.001 || abs(filtered.y-0.1) < 0.0004)){



		#ifdef CAVE_LIGHT_LEAK_FIX
			shading = mix(0.0, shading, clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0))*lightmap.y;
		#endif
		}
		


		vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));
		vec3 ambientLight = ambientUp*mix(clamp(ambientCoefs.y,0.,1.), 0.166, sssAmount);
		vec3 ambientLight2 = vec3(0.0);
		ambientLight += ambientDown*mix(clamp(-ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientRight*mix(clamp(ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientLeft*mix(clamp(-ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientB*mix(clamp(ambientCoefs.z,0.,1.), 0.166, sssAmount);
		ambientLight += ambientF*mix(clamp(-ambientCoefs.z,0.,1.), 0.166, sssAmount);

		vec3 custom_lightmap = texture2D(colortex4,(lightmap*10.0+0.5+vec2(0.0,19.))*texelSize).rgb*10./150./3.;
		

			#ifdef SSGI
			
			//	if (!hand)
				
					ambientLight2 = rtGI(normal, blueNoise(gl_FragCoord.xy), fragpos, ambientLight* custom_lightmap.x, sssAmount, custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*(vec3(TORCH_R,TORCH_G,TORCH_B)*(1+clamp(transparent.rgb,0,100))), normalize(albedo+1e-5)*0.7,luma(texture2D(colortex5,texcoord/RENDER_SCALE).rgb),ld(z),dataUnpacked1, edgemask, shadowCol);
			//	else
					
			if(hand)		ambientLight2 = ambientLight * custom_lightmap.x + custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B);
		
			#else
					ambientLight2 = ambientLight* custom_lightmap.x + custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B);
			#endif
			//combine all light sources
			


			

		//	gl_FragData[0].rgb = ambientLight2+((rsm*directLightCol.rgb*0.001)*lightmap.y);
			gl_FragData[0].rgb = ambientLight2;



				#ifndef SSGI

				float ao = 1.0;
				if (!hand)
					ssao(ao,fragpos,1.0,noise,worldToView(decode(dataUnpacked0.yw)),z);
				gl_FragData[0] *= ao;			
			#endif			
			gl_FragData[5].rgb = shadowCol.rgb ;
		
		
				gl_FragData[3].rgb = normal2 ;

				gl_FragData[3].a = texture2D(colortexA,texcoord).a ;


		

			#ifdef SPEC
			// Speculars
			// Unpack labpbr
			float roughness = unpackRoughness(trpData.x);
			float porosity = trpData.z;
			if (porosity > 64.5/255.0)
				porosity = 0.0;
			porosity = porosity*255.0/64.0;
			vec3 f0 = vec3(trpData.y);
			float rej = 1;


			if (f0.y > 229.5/255.0){
				f0 = albedo;
			}
		//	normal = texture2D(colortexA,texcoord).rgb;
			float rainMult = sqrt(lightmap.y)*wetness*(1.0-square(porosity));
			roughness = mix(roughness, 0.01, rainMult);
			f0 = mix(f0, vec3(0.02), rainMult);
			


			// Energy conservation between diffuse and specular
			vec3 fresnelDiffuse = vec3(0.0);

			// Sun specular
			vec3 specTerm = shading * GGX2(normal, -np3,  WsunVec, roughness+0.05*0.95, f0) * 8./150./3.;


			vec3 indirectSpecular = vec3(0.0);
			
			
			const int nSpecularSamples = SPEC_SSR_QUALITY;
			
			
			
			
			mat3 basis = CoordBase(normal);
			vec3 normSpaceView = -np3*basis;	
			float specstrength = SPECSTRENGTH;

			
		
			
			for (int i = 0; i < nSpecularSamples; i++){
			
			
				// Generate ray
				int seed = frameCounter*nSpecularSamples + i;
				vec2 ij = fract(R2_samples(seed) + blueNoise(gl_FragCoord.xy).rg);
				vec3 H = sampleGGXVNDF(normSpaceView, roughness, roughness, ij.x, ij.y);
				vec3 Ln = reflect(-normSpaceView, H);
				vec3 L = basis * Ln;

				// Ray contribution
				float g1 = g(clamp(dot(normal, L),0.0,1.0), roughness);
				vec3 F = f0 + (1.0 - f0) * pow(clamp(1.0 + dot(-Ln, H),0.0,1.0), 5.0);
				vec3 rayContrib = F * g1;
	
				// Skip calculations if ray does not contribute much to the lighting
			//	if (luma(rayContrib) > 0.02){
				if (luma(rayContrib) > 0.05){
					vec4 reflection = vec4(0.0,0.0,0.0,0.0);
					
					// Scale quality with ray contribution
					float rayQuality = 35*sqrt(luma(rayContrib));
						
					// Skip SSR if ray contribution is low
			//		if (rayQuality > 5.0) {
					if (rayQuality > 5.0) {
						vec3 rtPos = rayTrace(mat3(gbufferModelView) * L, fragpos.xyz, noise, rayQuality);
			
						// Reproject on previous frame
						if (rtPos.z < 1.){
							vec3 previousPosition = mat3(gbufferModelViewInverse) * toScreenSpace(rtPos) + gbufferModelViewInverse[3].xyz + cameraPosition-previousCameraPosition;
							previousPosition = mat3(gbufferPreviousModelView) * previousPosition + gbufferPreviousModelView[3].xyz;
							previousPosition.xy = projMAD(gbufferPreviousProjection, previousPosition).xy / -previousPosition.z * 0.5 + 0.5;
							
							if (previousPosition.x > 0.0 && previousPosition.y > 0.0 && previousPosition.x < 1.0 && previousPosition.x < 1.0) {
								reflection.a = 1.0;
								reflection.rgb = texture2D(colortex5,previousPosition.xy).rgb;

							}
						}
					}


					// Sample skybox
					if (reflection.a < 0.9){
						reflection.rgb = (skyCloudsFromTex(L, colortex4).rgb)*clamp(lightmap.y-0.8,0,1);
						reflection.rgb *= sqrt(lightmap.y)/150.*8./3.;
					}
					indirectSpecular += reflection.rgb * rayContrib;
					fresnelDiffuse += rayContrib;
				}

			}
		gl_FragData[5].rgb	= vec3(indirectSpecular);	

			
	vec3 speculars = (indirectSpecular/nSpecularSamples + specTerm * directLightCol.rgb)  ;
	vec3 closestToCamera = closestToCamera5taps(texcoord);
	vec3 fragposition = toScreenSpace(closestToCamera);			
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;
	previousPosition.xy = texcoord + velocity;

	vec3 albedoCurrent0 = texture2D(colortexE, texcoord).rgb;
	vec3 albedoCurrent1 = texture2D(colortexE, texcoord + vec2(texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent2 = texture2D(colortexE, texcoord + vec2(texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent3 = texture2D(colortexE, texcoord + vec2(-texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent4 = texture2D(colortexE, texcoord + vec2(-texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent5 = texture2D(colortexE, texcoord + vec2(0.0,texelSize.y)).rgb;
	vec3 albedoCurrent6 = texture2D(colortexE, texcoord + vec2(0.0,-texelSize.y)).rgb;
	vec3 albedoCurrent7 = texture2D(colortexE, texcoord + vec2(-texelSize.x,0.0)).rgb;
	vec3 albedoCurrent8 = texture2D(colortexE, texcoord + vec2(texelSize.x,0.0)).rgb;

	//Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each channel in the 3x3 neighborhood

	vec3 cMax = max(max(max(albedoCurrent0,albedoCurrent1),albedoCurrent2),max(albedoCurrent3,max(albedoCurrent4,max(albedoCurrent5,max(albedoCurrent6,max(albedoCurrent7,albedoCurrent8))))));
	vec3 cMin = min(min(min(albedoCurrent0,albedoCurrent1),albedoCurrent2),min(albedoCurrent3,min(albedoCurrent4,min(albedoCurrent5,min(albedoCurrent6,min(albedoCurrent7,albedoCurrent8))))));

	
			 
	vec3 albedoPrev = max(FastCatmulRom(colortexE, previousPosition.xy,vec4(texelSize, 1.0/texelSize), 0.0).xyz, 0.0);
	vec3 finalcAcc = clamp(albedoPrev,cMin,cMax);			
	
	float isclamped = clamp(clamp(((distance(albedoPrev,finalcAcc)/luma(albedoPrev))),0,100),0,100);	 


	

	 
	 rej = (isclamped)*clamp(length(velocity/texelSize),0.0,1.0) +isclamped ;


	 float weight = clamp( (rej+edgemask)  ,0.1,1.0);

		gl_FragData[4].rgb = mix(texture2D(colortexE, previousPosition.xy).rgb,(((indirectSpecular) /nSpecularSamples + specTerm * directLightCol.rgb)), weight );			
//		gl_FragData[4].rgb = (((indirectSpecular) /nSpecularSamples + specTerm * directLightCol.rgb));			
		
		
				
		


		#endif
		

		
	}

	
	
	
	
	
		
	
	

/* DRAWBUFFERS:8C9AE */
}
