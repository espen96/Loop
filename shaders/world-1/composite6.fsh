#version 150 compatibility
//Volumetric fog rendering
#extension GL_EXT_gpu_shader4 : enable

#define VL_SAMPLES 8 //[4 6 8 10 12 14 16 20 24 30 40 50]
#define Ambient_Mult 1.0 //[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.5 2.0 3.0 4.0 5.0 6.0 10.0]
#define SEA_LEVEL 70 //[0 10 20 30 40 50 60 70 80 90 100 110 120 130 150 170 190]	//The volumetric light uses an altitude-based fog density, this is where fog density is the highest, adjust this value according to your world.
#define ATMOSPHERIC_DENSITY 1.0 //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0 7.5 10.0 12.5 15.0 20.]
#define fog_mieg1 0.40 //[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0]
#define fog_mieg2 0.10 //[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0]
#define fog_coefficientRayleighR 5.8 //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0]
#define fog_coefficientRayleighG 1.35 //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0]
#define fog_coefficientRayleighB 3.31 //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0]

#define fog_coefficientMieR 2.0 //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0]
#define fog_coefficientMieG 5.0 //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0]
#define fog_coefficientMieB 10.0 //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0]

#define Underwater_Fog_Density 1.0 //[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.5 2.0 3.0 4.0]

flat in vec4 lightCol;
flat in vec3 ambientUp;
flat in vec3 ambientLeft;
flat in vec3 ambientRight;
flat in vec3 ambientB;
flat in vec3 ambientF;
flat in vec3 ambientDown;
flat in float tempOffsets;
flat in float fogAmount;
flat in float VFAmount;
uniform sampler2D noisetex;
uniform sampler2D depthtex0;


uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex4;

uniform vec3 sunVec;
uniform vec3 fogColor;
uniform float far;
uniform int frameCounter;
uniform float rainStrength;
uniform float sunElevation;
uniform ivec2 eyeBrightnessSmooth;
uniform float frameTimeCounter;
uniform int isEyeInWater;
uniform vec2 texelSize;
#include "lib/waterOptions.glsl"
#include "lib/color_transforms.glsl"
#include "lib/color_dither.glsl"
#include "lib/projections.glsl"
#include "lib/sky_gradient.glsl"
#include "/lib/res_params.glsl"
#include "/lib/noise.glsl"
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)



float phaseg(float x, float g){
    float gg = g * g;
    return (gg * -0.25 + 0.25) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5) /3.14;
}
float phaseRayleigh(float cosTheta) {
	const vec2 mul_add = vec2(0.1, 0.28) /acos(-1.0);
	return cosTheta * mul_add.x + mul_add.y; // optimized version from [Elek09], divided by 4 pi for energy conservation
}

float densityAtPos(in vec3 pos)
{

	pos /= 18.;
	pos.xz *= 0.5;


	vec3 p = floor(pos);
	vec3 f = fract(pos);

	f = (f*f) * (3.-2.*f);

	vec2 uv =  p.xz + f.xz + p.y * vec2(0.0,193.0);

	vec2 coord =  uv / 512.0;

	vec2 xy = texture(noisetex, coord).yx;

	return mix(xy.r,xy.g, f.y);
}
float cloudVol(in vec3 pos){

	vec3 samplePos = pos*vec3(1.0,1./32.,1.0)*5.0+frameTimeCounter*vec3(0.5,0.,0.5)*1.;
	float noise = densityAtPos(samplePos*12.);
	float unifCov = exp2(-max(pos.y-SEA_LEVEL,0.0)/50.);

	float cloud = pow(clamp(1.0-noise-0.76,0.0,1.0),2.)+0.005;

return cloud;
}
mat2x3 getVolumetricRays(float dither,vec3 fragpos) {

	//project pixel position into projected shadowmap space
	vec3 wpos = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;
	vec3 fragposition = mat3(shadowModelView) * wpos + shadowModelView[3].xyz;
	fragposition = diagonal3(shadowProjection) * fragposition + shadowProjection[3].xyz;



	//project view origin into projected shadowmap space
	vec3 start = toShadowSpaceProjected(vec3(0.));


	//rayvector into projected shadow map space
	//we can use a projected vector because its orthographic projection
	//however we still have to send it to curved shadow map space every step
	vec3 dV = (fragposition-start);
	vec3 dVWorld = (wpos-gbufferModelViewInverse[3].xyz);

	float maxLength = min(length(dVWorld),far)/length(dVWorld);
	dV *= maxLength;
	dVWorld *= maxLength;

	//apply dither
	vec3 progress = start.xyz;
	vec3 progressW = gbufferModelViewInverse[3].xyz+cameraPosition;
		vec3 vL = vec3(0.);

		float SdotV = dot(sunVec,normalize(fragpos))*lightCol.a;
		float dL = length(dVWorld);
		//Mie phase + somewhat simulates multiple scattering (Horizon zero down cloud approx)
		float mie = max(phaseg(SdotV,fog_mieg1),1.0/13.0);
		float rayL = phaseRayleigh(SdotV);
	//	wpos.y = clamp(wpos.y,0.0,1.0);

		vec3 ambientCoefs = dVWorld/dot(abs(dVWorld),vec3(1.));

		vec3 ambientLight = ambientUp;
		ambientLight += ambientDown;
		ambientLight += ambientRight;
		ambientLight += ambientLeft;
		ambientLight += ambientB;
		ambientLight += ambientF;

		vec3 skyCol0 = ambientLight*8.*2./150./3.*Ambient_Mult*3.1415;

		float mu = 1.0;
		float muS = 1.05;
		vec3 absorbance = vec3(1.0);
		float expFactor = 11.0;
	  vec3 fogColor = clamp(fogColor*pow(luma(fogColor),-0.75)*0.65,0.0,1.0)*0.05;
		for (int i=0;i<VL_SAMPLES+8;i++) {
			float d = (pow(expFactor, float(i+dither)/float(VL_SAMPLES+8))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
			float dd = pow(expFactor, float(i+dither)/float(VL_SAMPLES+8)) * log(expFactor) / float(VL_SAMPLES+8)/(expFactor-1.0);
			progressW = gbufferModelViewInverse[3].xyz+cameraPosition + d*dVWorld;
			float densityVol = cloudVol(progressW)*2.0;
			float density = densityVol;
			vec3 vL0 = density*fogColor;
			vL += (vL0 - vL0 * exp(-density*mu*dd*dL)) / (density*mu+0.00000001)*absorbance;
			absorbance *= clamp(exp(-density*mu*dd*dL),0.0,1.0);
		}
	return mat2x3(vL,absorbance);



}
float waterCaustics(vec3 wPos, vec3 lightSource){
	vec2 pos = (wPos.xz - lightSource.xz/lightSource.y*wPos.y)*4.0 ;
	vec2 movement = vec2(-0.02*frameTimeCounter);
	float caustic = 0.0;
	float weightSum = 0.0;
	float radiance =  2.39996;
	mat2 rotationMatrix  = mat2(vec2(cos(radiance),  -sin(radiance)),  vec2(sin(radiance),  cos(radiance)));
	vec2 displ = texture(noisetex, pos*vec2(3.0,1.0)/96. + movement).bb*2.0-1.0;
	pos = pos/2.+vec2(1.74*frameTimeCounter) ;
	for (int i = 0; i < 3; i++){
		pos = rotationMatrix * pos;
		caustic += pow(0.5+sin(dot(pos * exp2(0.8*i)+ displ*3.1415,vec2(0.5)))*0.5,6.0)*exp2(-0.8*i)/1.41;
		weightSum += exp2(-0.8*i);
	}
	return caustic * weightSum;
}
void waterVolumetrics(inout vec3 inColor, vec3 rayStart, vec3 rayEnd, float estEyeDepth, float estSunDepth, float rayLength, float dither, vec3 waterCoefs, vec3 scatterCoef, vec3 ambient, vec3 lightSource, float VdotL){
		int spCount = 16;

		vec3 start = toShadowSpaceProjected(rayStart);
		vec3 end = toShadowSpaceProjected(rayEnd);
		vec3 dV = (end-start);
		//limit ray length at 32 blocks for performance and reducing integration error
		//you can't see above this anyway
		float maxZ = min(rayLength,32.0)/(1e-8+rayLength);
		dV *= maxZ;
		rayLength *= maxZ;
		float dY = normalize(mat3(gbufferModelViewInverse) * rayEnd).y * rayLength;
		vec3 absorbance = vec3(1.0);
		vec3 vL = vec3(0.0);
		float phase = phaseg(VdotL, Dirt_Mie_Phase);
		float expFactor = 11.0;
		for (int i=0;i<spCount;i++) {
			float d = (pow(expFactor, float(i+dither)/float(spCount))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);		// exponential step position (0-1)
			float dd = pow(expFactor, float(i+dither)/float(spCount)) * log(expFactor) / float(spCount)/(expFactor-1.0);	//step length (derivative)
			vec3 spPos = start.xyz + dV*d;
			vec3 ambientMul = exp(-max(estEyeDepth - dY * d,0.0) * waterCoefs * 1.1);
			vec3 light = (ambientMul*ambient )*scatterCoef;
			vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs *absorbance;
			absorbance *= exp(-dd * rayLength * waterCoefs);
		}
		inColor += vL;
}

//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
/* RENDERTARGETS: 0 */
	if (isEyeInWater == 0){
		vec2 tc = floor(gl_FragCoord.xy)/VL_RENDER_RESOLUTION*texelSize+0.5*texelSize;
		float z = texture(depthtex0,tc).x;
		vec3 fragpos = toScreenSpace(vec3(tc/RENDER_SCALE,z));
		float noise=blueNoise();
		mat2x3 vl = getVolumetricRays(noise,fragpos);
		float absorbance = dot(vl[1],vec3(0.22,0.71,0.07));
		gl_FragData[0] = clamp(vec4(vl[0],absorbance),0.000001,65000.);
	}
	else {
		float dirtAmount = Dirt_Amount;
		vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
		vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
		vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
		vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B);
		vec2 tc = floor(gl_FragCoord.xy)/VL_RENDER_RESOLUTION*texelSize+0.5*texelSize;
		float z = texture(depthtex0,tc).x;
		vec3 fragpos = toScreenSpace(vec3(tc/RENDER_SCALE,z));
		float noise=blueNoise();
		vec3 vl = vec3(0.0);
		float estEyeDepth = clamp((14.0-eyeBrightnessSmooth.y/255.0*16.0)/14.0,0.,1.0);
		estEyeDepth *= estEyeDepth*estEyeDepth*34.0;
		#ifndef lightMapDepthEstimation
			estEyeDepth = max(Water_Top_Layer - cameraPosition.y,0.0);
		#endif
		waterVolumetrics(vl, vec3(0.0), fragpos, estEyeDepth, estEyeDepth, length(fragpos), noise, totEpsilon, scatterCoef, ambientUp*8./150./3.*0.5, lightCol.rgb*8./150./3.0*(1.0-pow(1.0-sunElevation*lightCol.a,5.0)), dot(normalize(fragpos), normalize(sunVec)));
		gl_FragData[0] = clamp(vec4(vl,1.0),0.000001,65000.);
	}

}
