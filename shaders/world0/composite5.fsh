#version 120
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
//#define POM

#include "/lib/res_params.glsl"
#define SSAO

#define ROUGHREF



#define Depth_Write_POM	// POM adjusts the actual position, so screen space shadows can cast shadows on POM
#define POM_DEPTH 0.15 // [0.025 0.05 0.075 0.1 0.125 0.15 0.20 0.25 0.30 0.50 0.75 1.0] //Increase to increase POM strength
#define CAVE_LIGHT_LEAK_FIX // Hackish way to remove sunlight incorrectly leaking into the caves. Can inacurrately create shadows in some places
//#define CLOUDS_SHADOWS
//#define LABSSS
#define CLOUDS_SHADOWS_STRENGTH 1.0 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]

#define SSPTMIX1 1.0 //[0.0 0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define SSPTambient 1.0 // [0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.20 0.22 0.23 0.25 0.27 0.29 0.32 0.34 0.37 0.40 0.43 0.46 0.50 0.54 0.58 0.63 0.68 0.74 0.79 0.86 0.93 1.00 1.08 1.17 1.26 1.36 1.47 1.59 1.71 1.85 2.00 2.16 2.33 2.51 2.72 2.93 3.17 3.42 3.69 3.99 4.30 4.65 5.02 5.42 5.85 6.32 6.82 7.37 7.95 8.59 9.27 10.01 10.81 11.68 12.61 13.61 14.70 15.87 17.14 18.51 19.99 21.58 23.30 25.16 27.17 29.34 31.68 34.21 36.94 39.89 43.07 46.50 50.22 54.22 58.55 63.22 68.27 73.72 79.60 85.95 92.81 100.22 108.21 116.85 126.17 136.24 147.11 158.85 171.53 185.22 200.00]

#define CLOUDS_QUALITY 0.35 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define TORCH_R 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_G 0.4 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_B 0.13 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Emissive_Strength 2.00 // [0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.20 0.22 0.23 0.25 0.27 0.29 0.32 0.34 0.37 0.40 0.43 0.46 0.50 0.54 0.58 0.63 0.68 0.74 0.79 0.86 0.93 1.00 1.08 1.17 1.26 1.36 1.47 1.59 1.71 1.85 2.00 2.16 2.33 2.51 2.72 2.93 3.17 3.42 3.69 3.99 4.30 4.65 5.02 5.42 5.85 6.32 6.82 7.37 7.95 8.59 9.27 10.01 10.81 11.68 12.61 13.61 14.70 15.87 17.14 18.51 19.99 21.58 23.30 25.16 27.17 29.34 31.68 34.21 36.94 39.89 43.07 46.50 50.22 54.22 58.55 63.22 68.27 73.72 79.60 85.95 92.81 100.22 108.21 116.85 126.17 136.24 147.11 158.85 171.53 185.22 200.00]


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
uniform sampler2D colortex2;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortexA;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;


uniform sampler2D colortexC;
uniform sampler2D colortexE;

uniform sampler2D colortex6; // Noise
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth



uniform sampler2DShadow shadow;

uniform sampler2DShadow shadowtex1;
uniform sampler2DShadow shadowtex0;
uniform sampler2DShadow shadowcolor0;
#define SPEC_SSR_QUALITY 2 //[1 2 3 4 5 6 7 8 9 10 ]


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
#include "/lib/util.glsl"


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


#include "/lib/waterOptions.glsl"
#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/stars.glsl"
#include "/lib/volumetricClouds.glsl"


float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}
#ifdef SPEC
uniform sampler2D colortexD;
#endif

#include "/lib/specular.glsl"
vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}
float lengthVec (vec3 vec){
	return sqrt(dot(vec,vec));
}

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
vec2 encode (vec3 unenc)
{
	unenc.xy = unenc.xy / dot(abs(unenc), vec3(1.0)) + 0.00390625;
	unenc.xy = unenc.z <= 0.0 ? (1.0 - abs(unenc.yx)) * sign(unenc.xy) : unenc.xy;
    vec2 encn = unenc.xy * 0.5 + 0.5;
	
    return vec2((encn));
}

float rayTraceShadow(vec3 dir,vec3 position,float dither){

    const float quality = 32.;
    vec3 clipPosition = toClipSpace3(position);
	//prevents the ray from going behind the camera
	float rayLength = ((position.z + dir.z * far*sqrt(3.)) > -near) ?
       (-near -position.z) / dir.z : far*sqrt(3.);
    vec3 direction = toClipSpace3(position+dir*rayLength)-clipPosition;  //convert to clip space
    direction.xyz = direction.xyz/max(abs(direction.x)/texelSize.x,abs(direction.y)/texelSize.y);	//fixed step size




    vec3 stepv = direction *3. * clamp(MC_RENDER_QUALITY,1.,2.0)*vec3(RENDER_SCALE,1.0);

	vec3 spos = clipPosition*vec3(RENDER_SCALE,1.0)+vec3(TAA_Offset*vec2(texelSize.x,texelSize.y)*0.5,0.0)+stepv*dither;





	for (int i = 0; i < int(quality); i++) {
		spos += stepv;

		float sp = texture2D(depthtex1,spos.xy).x;
        if( sp < spos.z) {

			float dist = abs(linZ(sp)-linZ(spos.z))/linZ(spos.z);

			if (dist < 0.01 ) return 0.0;



	}

	}
    return 1.0;
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

float waterCaustics(vec3 wPos, vec3 lightSource){
	vec2 pos = (wPos.xz - lightSource.xz/lightSource.y*wPos.y)*4.0 ;
	vec2 movement = vec2(-0.02*frameTimeCounter);
	float caustic = 0.0;
	float weightSum = 0.0;
	float radiance =  2.39996;
	mat2 rotationMatrix  = mat2(vec2(cos(radiance),  -sin(radiance)),  vec2(sin(radiance),  cos(radiance)));
	vec2 displ = texture2D(noisetex, pos*vec2(3.0,1.0)/96. + movement).bb*2.0-1.0;
	pos = pos*0.5+vec2(1.74*frameTimeCounter) ;
	for (int i = 0; i < 3; i++){
		pos = rotationMatrix * pos;
		caustic += Pow6(0.5+sin(dot(pos * exp2(0.8*i)+ displ*3.1415,vec2(0.5)))*0.5)*exp2(-0.8*i)/1.41;
		weightSum += exp2(-0.8*i);
	}
	return caustic * weightSum;
}

void waterVolumetrics(inout vec3 inColor, vec3 rayStart, vec3 rayEnd, float estEndDepth, float estSunDepth, float rayLength, float dither, vec3 waterCoefs, vec3 scatterCoef, vec3 ambient, vec3 lightSource, float VdotL){
		inColor *= exp(-rayLength * waterCoefs);	//No need to take the integrated value
		int spCount = rayMarchSampleCount;
		vec3 start = toShadowSpaceProjected(rayStart);
		vec3 end = toShadowSpaceProjected(rayEnd);
		vec3 dV = (end-start);
		//limit ray length at 32 blocks for performance and reducing integration error
		//you can't see above this anyway
		float maxZ = min(rayLength,32.0)/(1e-8+rayLength);
		dV *= maxZ;
		vec3 dVWorld = -mat3(gbufferModelViewInverse) * (rayEnd - rayStart) * maxZ;
		rayLength *= maxZ;
		estEndDepth *= maxZ;
		estSunDepth *= maxZ;
		vec3 absorbance = vec3(1.0);
		vec3 vL = vec3(0.0);
		float phase = phaseg(VdotL, Dirt_Mie_Phase);
		float expFactor = 11.0;
		vec3 progressW = gbufferModelViewInverse[3].xyz+cameraPosition;
		for (int i=0;i<spCount;i++) {
			float d = (pow(expFactor, float(i+dither)/float(spCount))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
			float dd = pow(expFactor, float(i+dither)/float(spCount)) * log(expFactor) / float(spCount)/(expFactor-1.0);
			vec3 spPos = start.xyz + dV*d;
			progressW = gbufferModelViewInverse[3].xyz+cameraPosition + d*dVWorld;
			//project into biased shadowmap space
			float distortFactor = calcDistort(spPos.xy);
			vec3 pos = vec3(spPos.xy*distortFactor, spPos.z);
			float sh = 1.0;
			if (abs(pos.x) < 1.0-0.5/2048. && abs(pos.y) < 1.0-0.5/2048){
				pos = pos*vec3(0.5,0.5,0.5*0.166)+0.5;
				sh =  shadow2D( shadow, pos).x;
			}
			vec3 ambientMul = exp(-estEndDepth * d * waterCoefs * 1.1);
			vec3 sunMul = exp(-estSunDepth * d * waterCoefs);
			vec3 light = (sh * lightSource*8./150./3.0 * phase * sunMul + ambientMul * ambient)*scatterCoef;
			vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs *absorbance;
			absorbance *= exp(-dd * rayLength * waterCoefs);
		}
		inColor += vL;
}
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
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

vec3 closestToCamera5taps(vec2 texcoord)
{
	vec2 du = vec2(texelSize.x*2., 0.0);
	vec2 dv = vec2(0.0, texelSize.y*2.);

	vec3 dtl = vec3(texcoord,0.) + vec3(-texelSize, texture2D(depthtex0, texcoord - dv - du).x);
	vec3 dtr = vec3(texcoord,0.) +  vec3( texelSize.x, -texelSize.y, texture2D(depthtex0, texcoord - dv + du).x);
	vec3 dmc = vec3(texcoord,0.) + vec3( 0.0, 0.0, texture2D(depthtex0, texcoord).x);
	vec3 dbl = vec3(texcoord,0.) + vec3(-texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv - du).x);
	vec3 dbr = vec3(texcoord,0.) + vec3( texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv + du).x);

	vec3 dmin = dmc;
	dmin = dmin.z > dtr.z? dtr : dmin;
	dmin = dmin.z > dtl.z? dtl : dmin;
	dmin = dmin.z > dbl.z? dbl : dmin;
	dmin = dmin.z > dbr.z? dbr : dmin;
	#ifdef TAA_UPSCALING
	dmin.xy = dmin.xy/RENDER_SCALE;
	#endif
	return dmin;
}
vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
vec3 FastCatmulRom(sampler2D colorTex, vec2 texcoord, vec4 rtMetrics, float sharpenAmount)
{
    vec2 position = rtMetrics.zw * texcoord;
    vec2 centerPosition = floor(position - 0.5) + 0.5;
    vec2 f = position - centerPosition;
    vec2 f2 = f * f;
    vec2 f3 = f * f2;

    float c = sharpenAmount;
    vec2 w0 =        -c  * f3 +  2.0 * c         * f2 - c * f;
    vec2 w1 =  (2.0 - c) * f3 - (3.0 - c)        * f2         + 1.0;
    vec2 w2 = -(2.0 - c) * f3 + (3.0 -  2.0 * c) * f2 + c * f;
    vec2 w3 =         c  * f3 -                c * f2;

    vec2 w12 = w1 + w2;
    vec2 tc12 = rtMetrics.xy * (centerPosition + w2 / w12);
    vec3 centerColor = texture2D(colorTex, vec2(tc12.x, tc12.y)).rgb;

    vec2 tc0 = rtMetrics.xy * (centerPosition - 1.0);
    vec2 tc3 = rtMetrics.xy * (centerPosition + 2.0);
    vec4 color = vec4(texture2D(colorTex, vec2(tc12.x, tc0.y )).rgb, 1.0) * (w12.x * w0.y ) +
                   vec4(texture2D(colorTex, vec2(tc0.x,  tc12.y)).rgb, 1.0) * (w0.x  * w12.y) +
                   vec4(centerColor,                                      1.0) * (w12.x * w12.y) +
                   vec4(texture2D(colorTex, vec2(tc3.x,  tc12.y)).rgb, 1.0) * (w3.x  * w12.y) +
                   vec4(texture2D(colorTex, vec2(tc12.x, tc3.y )).rgb, 1.0) * (w12.x * w3.y );
	return color.rgb/color.a;

}
vec3 tonemap(vec3 col){
	return col/(1+luma(col));
}
vec3 invTonemap(vec3 col){
	return col/(1-luma(col));
}		
//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 unenc, vec2 lightmaps)
{
	unenc.xy = unenc.xy / dot(abs(unenc), vec3(1.0)) + 0.00390625;
	unenc.xy = unenc.z <= 0.0 ? (1.0 - abs(unenc.yx)) * sign(unenc.xy) : unenc.xy;
    vec2 encn = unenc.xy * 0.5 + 0.5;
	
    return vec4((encn),vec2(lightmaps.x,lightmaps.y));
}


void main() {
	vec2 texcoord = gl_FragCoord.xy*texelSize;
	float dirtAmount = Dirt_Amount;
	vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
	vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
	vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
	vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B);
	float z0 = texture2D(depthtex0,texcoord).x;
	float z = texture2D(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();

	vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normalize(p3);

	
	
	
	
	//sky
	if (z >=1.0) {
		vec3 color = vec3(0.0);
		vec4 cloud = texture2D_bicubic(colortex0,texcoord*CLOUDS_QUALITY);
		if (np3.y > 0.){
			color += stars(np3);
			color += drawSun(dot(lightCol.a*WsunVec,np3),0, lightCol.rgb/150.,vec3(0.0));
		}
		color += skyFromTex(np3,colortex4)/150. + toLinear(texture2D(colortex1,texcoord).rgb)/10.*4.0*ffstep(0.985,-dot(lightCol.a*WsunVec,np3));
		color = color*cloud.a+cloud.rgb;
		gl_FragData[0].rgb = clamp(fp10Dither(color*8./3.,triangularize(noise)),0.0,65000.);
		//if (gl_FragData[0].r > 65000.) 	gl_FragData[0].rgb = vec3(0.0);
		vec4 trpData = texture2D(colortex7,texcoord);
		bool iswater = texture2D(colortex7,texcoord).a > 0.99;
		if (iswater){
			vec3 fragpos0 = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z0));
			float Vdiff = distance(fragpos,fragpos0);
			float VdotU = np3.y;
			float estimatedDepth = Vdiff * abs(VdotU);	//assuming water plane
			float estimatedSunDepth = estimatedDepth/abs(refractedSunVec.y); //assuming water plane

			vec3 lightColVol = lightCol.rgb * (1.0-Pow5(1.0-WsunVec.y));	//fresnel
			vec3 ambientColVol = ambientUp*8./150./3.*0.5 * eyeBrightnessSmooth.y / 240.0;
			if (isEyeInWater == 0)
				waterVolumetrics(gl_FragData[0].rgb, fragpos0, fragpos, estimatedDepth, estimatedSunDepth, Vdiff, noise, totEpsilon, scatterCoef, ambientColVol, lightColVol, dot(np3, WsunVec));
		}
	}
	//land
	else {
		p3 += gbufferModelViewInverse[3].xyz;

		vec4 trpData = texture2D(colortex7,texcoord);
		vec4 transparent = texture2D(colortex2,texcoord);
		
		bool iswater = texture2D(colortex7,texcoord).a > 0.99;

		vec4 mask = texture2D(colortex9,texcoord);
		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		
		
		vec4 shadowCol = vec4(0.0);
		float caustics = 1;
		
		
		vec3 normal = mat3(gbufferModelViewInverse) * worldToView(decode(dataUnpacked0.yw));


		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool entity = abs(mask.r) >0.9;
		vec2 lightmap = dataUnpacked1.yz;
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;	// Strong translucency
		bool translucent2 = abs(dataUnpacked1.w-0.6) <0.01;	// Weak translucency
		float labsss = trpData.z;


			if (labsss < 64.5/255.0)
				labsss = 0.0;

		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		float NdotLGeom = dot(normal, WsunVec);
		float NdotL = NdotLGeom;
		if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater == 1))
			NdotL = dot(normal, refractedSunVec);

		float diffuseSun = clamp(NdotL,0.,1.0);
		vec3 filtered = vec3(1.412,1.0,0.0);
		vec3 caustic = vec3(1.412,1.0,0.0);
		if (!hand){
			filtered = texture2D(colortex3,texcoord).rgb;
		}
		float shading = 1.0 - filtered.b;
		float pShadow = filtered.b*2.0-1.0;

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
				float rdMulcaustic = caustic.x*distortFactor*d0*k/shadowMapResolution*2;
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

					float isShadow = shadow2D(shadow,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x;
				//	float isShadow =shadow2D(shadowtex1,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x;


		
	


				//	vec2 offsetS = tapLocation(i,2/2, 0.0,noise,0.0);

				//	float weight = 1.0+(i)*rdMul/2*shadowMapResolution;
					
				
						float shadow1 = shadow2D(shadowtex1,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x;
						float shadow0 = shadow2D(shadowtex0,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x;
							 shadowCol = shadow2D(shadowcolor0,vec3(projectedShadowPosition + vec3(rdMul*offsetS*(Pow2(filtered.x)*0.25),-diffthresh*weight))).xyzw;
						float transparentshadow = (shadow1-shadow0);	 
							 shadowCol = shadowCol *transparentshadow;
							 shadowCol.rgb = (shadowCol.rgb * (1-(shadowCol.a*0.5)))*0.5;
							transparentshadow -= shadowCol.a;
		
						shading += clamp((isShadow+transparentshadow)/SHADOW_FILTER_SAMPLE_COUNT,0,1);
						
						
									gl_FragData[1].rgb = vec3(shadowCol);
				}
			}
		}
		//custom shading model for translucent objects
		#ifdef Variable_Penumbra_Shadows
		#ifndef LABSSS
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
		
		#else
			if (labsss > 0.0) {
			sssAmount = labsss;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			diffuseSun *= 1.0 - sssAmount;
			SSS *= clamp(sqrt(lightmap.y*2-1.5),0,1);
			SSS *= shading;
		}	
		#endif
		#endif


		if ((diffuseSun*shading > 0.001 || abs(filtered.y-0.1) < 0.0004)){

		if(!hand){	
		#ifdef SCREENSPACE_CONTACT_SHADOWS
				vec3 vec = lightCol.a*sunVec;
				float screenShadow = rayTraceShadow(vec,fragpos,noise);
				shading = min(screenShadow, shading);
				// Out of shadow map
				if (abs(filtered.y-0.1) < 0.0004)
					SSS *= 1;
			#endif 
			}

		#ifdef CAVE_LIGHT_LEAK_FIX
			shading = mix(0.0, shading, clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0))*lightmap.y;
			shadowCol.rgb = mix(vec3(0.0), shadowCol.rgb, clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0))*lightmap.y;
		#endif
		}
		#ifdef CLOUDS_SHADOWS
			vec3 pos = p3 + cameraPosition;
			const int rayMarchSteps = 6;
			float cloudShadow = 0.0;
			for (int i = 0; i < rayMarchSteps; i++){
				vec3 cloudPos = pos + WsunVec/abs(WsunVec.y)*(1500+(noise+i)/rayMarchSteps*1700-pos.y);
				cloudShadow += getCloudDensity(cloudPos, 0);
			}
			cloudShadow = mix(1.0,exp(-cloudShadow*cloudDensity*1700/rayMarchSteps),mix(CLOUDS_SHADOWS_STRENGTH,1.0,rainStrength));
			shading *= cloudShadow;
			SSS *= cloudShadow;
		#endif

		vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));
		vec3 ambientLight = ambientUp*mix(clamp(ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientDown*mix(clamp(-ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientRight*mix(clamp(ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientLeft*mix(clamp(-ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientB*mix(clamp(ambientCoefs.z,0.,1.), 0.166, sssAmount);
		ambientLight += ambientF*mix(clamp(-ambientCoefs.z,0.,1.), 0.166, sssAmount);

		vec3 directLightCol = lightCol.rgb;
		vec3 custom_lightmap = texture2D(colortex4,(lightmap*15.0+0.5+vec2(0.0,19.))*texelSize).rgb*10./150./3.;
		float emitting = 0.0;
		if (emissive || (hand && heldBlockLightValue > 0.1)){
		if(!hand)	emitting = (luma(albedo)*8.0*Emissive_Strength)*2-1;
		if (hand)   emitting = (luma(albedo)*Emissive_Strength)*2-1;
			custom_lightmap.y = 0.0;
			emitting = clamp(emitting*(1-luma(transparent.rgb*20)),0.0,10);
		}
		if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater == 1)){
			vec3 fragpos0 = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z0));
			float Vdiff = distance(fragpos,fragpos0);
			float VdotU = np3.y;
			float estimatedDepth = Vdiff * abs(VdotU);	//assuming water plane
			if (isEyeInWater == 1){
				Vdiff = length(fragpos);
				estimatedDepth =  clamp((15.5-lightmap.y*16.0)/15.5,0.,1.0);
				estimatedDepth *= estimatedDepth*estimatedDepth*32.0;
				#ifndef lightMapDepthEstimation
					estimatedDepth = max(Water_Top_Layer - (cameraPosition.y+p3.y),0.0);
				#endif
			}
			float estimatedSunDepth = estimatedDepth/abs(refractedSunVec.y); //assuming water plane
			directLightCol *= exp(-totEpsilon*estimatedSunDepth)*(1.0-Pow5(1.0-WsunVec.y));
			float caustics = waterCaustics(mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition, refractedSunVec);
			directLightCol *= mix(caustics*0.5+0.5,1.0,exp(-estimatedSunDepth/3.0));

			if (isEyeInWater == 0){
				ambientLight *= min(exp(-totEpsilon*estimatedDepth), custom_lightmap.x);
				ambientLight += custom_lightmap.z;
			}
			else {
				ambientLight += 10.0 * exp(-totEpsilon*8.0);
				ambientLight *= exp(-totEpsilon*estimatedDepth)*8./150./3.;
			}
			ambientLight *= mix(caustics,1.0,0.85);
			ambientLight += custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B);

			//combine all light sources

			gl_FragData[0].rgb = ((shading*diffuseSun + SSS)/pi*8./150./3.*directLightCol.rgb + ambientLight + emitting) * albedo;
			//Bruteforce integration is probably overkill
			vec3 lightColVol = lightCol.rgb * (1.0-Pow5(1.0-WsunVec.y));	//fresnel
			vec3 ambientColVol =  ambientUp*8./150./3.*0.5 / 240.0 * eyeBrightnessSmooth.y;
			if (isEyeInWater == 0)
				waterVolumetrics(gl_FragData[0].rgb, fragpos0, fragpos, estimatedDepth, estimatedSunDepth, Vdiff, noise, totEpsilon, scatterCoef, ambientColVol, lightColVol, dot(np3, WsunVec));

		}
		else {


				



			ambientLight = texture2D(colortex8,texcoord).rgb;
			ambientLight *= (1+clamp(transparent.rgb*10*emitting*2,1,100));					
					

		float labemissive = texture2D(colortexA, texcoord).a;
		ambientLight += vec3(labemissive);
		
	


				//	if(hand) shading = 1.0;
					

			//combine all light sources
			caustic = shadowCol.rgb ;			
			

		//	gl_FragData[0].rgb = (((shading * diffuseSun + (caustic*0.2) + SSS )/pi*8./150./3.*directLightCol.rgb + ambientLight + emitting)*albedo) ;
			gl_FragData[0].rgb = (((shading * diffuseSun + (caustic*0.2) + SSS )/pi*8./150./3.*directLightCol.rgb + ambientLight + emitting)*albedo) ;



			
			// Speculars
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

			float rainMult = sqrt(lightmap.y)*wetness*(1.0-square(porosity));
			roughness = mix(roughness, 0.01, rainMult);
			f0 = mix(f0, vec3(0.02), rainMult);
			


			// Energy conservation between diffuse and specular
			vec3 fresnelDiffuse = vec3(0.0);

			// Sun specular
			vec3 indirectSpecular = vec3(0.0);
			
			
			const int nSpecularSamples = SPEC_SSR_QUALITY;
			

			mat3 basis = CoordBase(normal);
			vec3 normSpaceView = -np3*basis;	


			
		
			
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
				if (luma(rayContrib) > 0.02){



					// Sample skybox

					fresnelDiffuse += rayContrib;
				}

			}


		indirectSpecular.rgb =  texture2D(colortexE,texcoord).rgb;

			

			


		if (!hand) 			gl_FragData[0].rgb = indirectSpecular +  (  (1.0-fresnelDiffuse/nSpecularSamples*lightmap.x) * gl_FragData[0].rgb );



		#endif

		}

	}




	
	


/* DRAWBUFFERS:3D */
}
