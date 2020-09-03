

#version 120
//Volumetric fog rendering


#include "/lib/settings.glsl"

#extension GL_EXT_gpu_shader4 : enable

flat varying vec4 lightCol;
flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;
flat varying float tempOffsets;
flat varying float fogAmount;
flat varying float VFAmount;
uniform sampler2D noisetex;
uniform sampler2D depthtex0;
uniform sampler2DShadow shadow;
uniform float blindness; 

uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D texture;
uniform vec3 fogColor; 

uniform float fogDensity;
uniform vec3 sunVec;
uniform float far;
uniform int frameCounter;
uniform float rainStrength;
uniform float sunElevation;
uniform ivec2 eyeBrightnessSmooth;
uniform float frameTimeCounter;
uniform int isEyeInWater;
uniform vec2 texelSize;
varying vec2 texcoord;
varying vec4 color;					  
#include "/lib/waterOptions.glsl"
#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/color_dither.glsl"
#include "/lib/projections.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/res_params.glsl"							   
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)

#ifdef VOLUMETRIC_FOG					 
float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+tempOffsets);
}


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

	vec2 xy = texture2D(noisetex, coord).yx;

	return mix(xy.r,xy.g, f.y);
}
float cloudVol(in vec3 pos){

	vec3 samplePos = pos*vec3(1.0,1./16.,1.0)+frameTimeCounter*vec3(0.5,0.,0.5)*5.;
	float coverage = mix(exp2(-(pos.y-SEA_LEVEL)*(pos.y-SEA_LEVEL)/10000.),1.0,rainStrength*0.5);
	float noise = densityAtPos(samplePos*13.);
	float unifCov = exp2(-max(pos.y-SEA_LEVEL,0.0)/50.);

	float cloud = pow(clamp(coverage-noise-0.76,0.0,1.0),2.)*1200./0.23/(coverage+0.01)*VFAmount*600+unifCov*60.*fogAmount+rainStrength*2.;

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
	#ifdef TOASTER
	      maxLength = min(length(dVWorld),far*0.712)/length(dVWorld);
	#endif
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


		vec3 ambientCoefs = dVWorld/dot(abs(dVWorld),vec3(1.));

		vec3 ambientLight = ambientUp*clamp(ambientCoefs.y,0.,1.);
		ambientLight += ambientDown*clamp(-ambientCoefs.y,0.,1.);
		ambientLight += ambientRight*clamp(ambientCoefs.x,0.,1.);
		ambientLight += ambientLeft*clamp(-ambientCoefs.x,0.,1.);
		ambientLight += ambientB*clamp(ambientCoefs.z,0.,1.);
		ambientLight += ambientF*clamp(-ambientCoefs.z,0.,1.);

		vec3 skyCol0 = ambientLight*eyeBrightnessSmooth.y/vec3(240.)*Ambient_Mult*4.0/pi*8./150./3.;
		// Makes fog more white idk how to simulate it correctly
		vec3 sunColor = mix(lightCol.rgb, vec3(luma(lightCol.rgb)), 0.45) *8./150./3.;
		skyCol0 = mix(skyCol0.rgb, vec3(luma(skyCol0.rgb)), 0.45);

		vec3 rC = vec3(fog_coefficientRayleighR*1e-6, fog_coefficientRayleighG*1e-5, fog_coefficientRayleighB*1e-5);
		vec3 mC = vec3(fog_coefficientMieR*1e-6, fog_coefficientMieG*1e-6, fog_coefficientMieB*1e-6);


		float mu = 1.0;
		float muS = 1.0*mu;
		vec3 absorbance = vec3(1.0);
		float expFactor = 11.0;
		for (int i=0;i<N_VL_SAMPLES;i++) {
			float d = (pow(expFactor, float(i+dither)/float(N_VL_SAMPLES))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
			float dd = pow(expFactor, float(i+dither)/float(N_VL_SAMPLES)) * log(expFactor) / float(N_VL_SAMPLES)/(expFactor-1.0);
			progress = start.xyz + d*dV;
			progressW = gbufferModelViewInverse[3].xyz+cameraPosition + d*dVWorld;
			//project into biased shadowmap space
			float distortFactor = calcDistort(progress.xy);
			vec3 pos = vec3(progress.xy*distortFactor, progress.z);
			float densityVol = cloudVol(progressW)+(100*blindness)+(100*fogDensity);
			float sh = 1.0;
			if (abs(pos.x) < 1.0-0.5/2048. && abs(pos.y) < 1.0-0.5/2048){
				pos = pos*vec3(0.5,0.5,0.5/6.0)+0.5;
				sh =  shadow2D( shadow, pos).x;
			}
			//Water droplets(fog)
			float density = densityVol*ATMOSPHERIC_DENSITY*mu*500.0;
			#ifdef TOASTER
				  density = densityVol*ATMOSPHERIC_DENSITY*mu*1000.;
			#endif
			//Just air
			vec2 airCoef = exp2(-max(progressW.y-SEA_LEVEL,0.0)/vec2(8.0e3, 1.2e3)*vec2(6.,7.0))*6.0;

			//Pbr for air, yolo mix between mie and rayleigh for water droplets
			vec3 rL = rC*airCoef.x;
			vec3 m = (airCoef.y+density)*mC;
			vec3 vL0 = sunColor*sh*(rayL*rL+m*mie) + skyCol0*(rL+m);
		//	     vL0 += (fogColor*2-0.85)/50;

				 
			vL += (vL0 - vL0 * exp(-(rL+m)*dd*dL)) / ((rL+m)+0.00000001)*absorbance;
			absorbance *= clamp(exp(-(rL+m)*dd*dL),0.0,1.0);
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
	for (int i = 0; i < 5; i++){
		vec2 displ = texture2D(noisetex, pos/32.0 + movement).bb*2.0-1.0;
		pos = rotationMatrix * pos;
		caustic += pow(0.5+sin(dot((pos+vec2(1.74*frameTimeCounter)) * exp2(0.8*i) + displ*3.0,vec2(0.5)))*0.5,6.0)*exp2(-0.8*i)/1.41;
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
		vec3 dVWorld = mat3(gbufferModelViewInverse) * (rayEnd - rayStart) * maxZ;
		rayLength *= maxZ;
		float dY = normalize(mat3(gbufferModelViewInverse) * rayEnd).y * rayLength;
		vec3 absorbance = vec3(1.0);
		vec3 vL = vec3(0.0);
		float phase = phaseg(VdotL, Dirt_Mie_Phase);
		float expFactor = 11.0;
		vec3 progressW = gbufferModelViewInverse[3].xyz+cameraPosition;
		vec3 WsunVec = mat3(gbufferModelViewInverse) * sunVec;		
		for (int i=0;i<spCount;i++) {
			float d = (pow(expFactor, float(i+dither)/float(spCount))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);		// exponential step position (0-1)
			float dd = pow(expFactor, float(i+dither)/float(spCount)) * log(expFactor) / float(spCount)/(expFactor-1.0);	//step length (derivative)
			vec3 spPos = start.xyz + dV*d;
			progressW = gbufferModelViewInverse[3].xyz+cameraPosition + d*dVWorld;
			//project into biased shadowmap space
			float distortFactor = calcDistort(spPos.xy);
			vec3 pos = vec3(spPos.xy*distortFactor, spPos.z);
			float sh = 1.0;
			if (abs(pos.x) < 1.0-0.5/2048. && abs(pos.y) < 1.0-0.5/2048){
				pos = pos*vec3(0.5,0.5,0.5/6.0)+0.5;
				sh =  shadow2D( shadow, pos).x;
			}
			vec3 ambientMul = exp(-max(estEyeDepth - dY * d,0.0) * waterCoefs * 1.1);
			vec3 sunMul = exp(-max((estEyeDepth - dY * d) ,0.0)/abs(sunElevation) * waterCoefs);
			float sunCaustics = mix(waterCaustics(progressW, WsunVec)*0.5+0.5,1.0,exp(-max((estEyeDepth - dY * d) ,0.0)/3.0));
			vec3 light = (sh * sunCaustics * lightSource * phase * sunMul + (waterCaustics(progressW, vec3(0,1,0))*0.15+0.85)*ambientMul*ambient )*scatterCoef;
			vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs *absorbance;
			absorbance *= exp(-dd * rayLength * waterCoefs);
		}
		inColor += vL;
}
float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}
#endif	  
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
/* DRAWBUFFERS:0 */


		   
	vec2 texcoord = gl_FragCoord.xy*texelSize;							


	#ifndef VOLUMETRIC_FOG
	gl_FragData[0] = color*1-blindness;
	#endif			

	
	if (isEyeInWater == 0){

	#ifdef VOLUMETRIC_FOG									  
		vec2 tc = floor(gl_FragCoord.xy)/VL_RENDER_RESOLUTION*texelSize+0.5*texelSize;
		float z = texture2D(depthtex0,tc).x;
		vec3 fragpos = toScreenSpace(vec3(tc/RENDER_SCALE,z));
		float noise=blueNoise();
		mat2x3 vl = getVolumetricRays(noise,fragpos);
		float absorbance = dot(vl[1],vec3(0.22,0.71,0.07))-blindness;
		gl_FragData[0] = clamp(vec4(vl[0],absorbance),0.000001,65000.);
		vec4 trpData = texture2D(colortex3,texcoord);
		vec3 mask2 =vec3(0,0,0);
		if(trpData.a > 0.2 && trpData.a <0.9) mask2=vec3(0,0,1);
	#endif  
	}
	
	
	else {

		float estEyeDepth = clamp((14.0-eyeBrightnessSmooth.y/255.0*16.0)/14.0,0.,1.0);
		estEyeDepth *= estEyeDepth*estEyeDepth*34.0;
		#ifndef lightMapDepthEstimation
			estEyeDepth = max(Water_Top_Layer - cameraPosition.y,0.0);
		#endif

		#ifdef VOLUMETRIC_FOG
		float dirtAmount = Dirt_Amount;
		vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
		vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
		vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
		vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B) / pi;
		vec2 tc = floor(gl_FragCoord.xy)/VL_RENDER_RESOLUTION*texelSize+0.5*texelSize;
		float z = texture2D(depthtex0,tc).x;
		vec3 fragpos = toScreenSpace(vec3(tc/RENDER_SCALE,z));
		float noise=blueNoise();
		vec3 vl = vec3(0.0);
																				 
											  
								 
															 
		
		waterVolumetrics(vl, vec3(0.0), fragpos, estEyeDepth, estEyeDepth, length(fragpos), noise, totEpsilon, scatterCoef, ambientUp*8./150./3.*0.84*2.0/pi, lightCol.rgb*8./150./3.0*(0.91-pow(1.0-sunElevation,5.0)*0.86), dot(normalize(fragpos), normalize(sunVec)));
		gl_FragData[0] = clamp(vec4(vl,1.0),0.000001,65000.)*1-(blindness*0.95);

		#endif
		}	
	}
