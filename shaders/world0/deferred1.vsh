#version 120
#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"


flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;
flat varying vec3 zenithColor;
flat varying vec3 sunColor;
flat varying vec3 sunColorCloud;
flat varying vec3 moonColor;
flat varying vec3 moonColorCloud;
flat varying vec3 lightSourceColor;
flat varying vec3 avgSky;
flat varying vec2 tempOffsets;
flat varying float exposure;
flat varying float avgBrightness;
flat varying float exposureF;
flat varying float rodExposure;
flat varying float fogAmount;
flat varying float VFAmount;
flat varying float avgL2;

uniform sampler2D colortex4;
uniform sampler2D colortex6;

uniform mat4 gbufferModelViewInverse;
uniform vec3 sunPosition;
uniform vec2 texelSize;
uniform float rainStrength;
uniform float sunElevation;
uniform float nightVision;
uniform float frameTime;
uniform float eyeAltitude;
uniform int frameCounter;
uniform int worldTime;
vec3 sunVec = normalize(mat3(gbufferModelViewInverse) *sunPosition);



#include "/lib/sky_gradient.glsl"
#include "/lib/util.glsl"
#include "/lib/ROBOBO_sky.glsl"
vec3 rodSample(vec2 Xi)
{
	float r = sqrt(1.0f - Xi.x*Xi.y);
    float phi = 2 * 3.14159265359 * Xi.y;

    return normalize(vec3(cos(phi) * r, sin(phi) * r, Xi.x)).xzy;
}


float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}


//Low discrepancy 2D sequence, integration error is as low as sobol but easier to compute : http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}
float tanh(float x){
	return (exp(x) - exp(-x))/(exp(x) + exp(-x));
}
void main() {

	gl_Position = ftransform()*0.5+0.5;
	gl_Position.xy = gl_Position.xy*vec2(18.+258*2,258.)*texelSize;
	gl_Position.xy = gl_Position.xy*2.-1.0;

	tempOffsets = R2_samples(frameCounter%10000);

	ambientUp = vec3(0.0);
	ambientDown = vec3(0.0);
	ambientLeft = vec3(0.0);
	ambientRight = vec3(0.0);
	ambientB = vec3(0.0);
	ambientF = vec3(0.0);
	avgSky = vec3(0.0);
	//Integrate sky light for each block side
	int maxIT = 20;
	for (int i = 0; i < maxIT; i++) {
			vec2 ij = R2_samples((frameCounter%1000)*maxIT+i);
			vec3 pos = normalize(rodSample(ij));


			vec3 samplee = 2.0*skyFromTex(pos,colortex4).rgb/maxIT/150.;
			avgSky += samplee/2.0;
			ambientUp += samplee*(pos.y+abs(pos.x)/7.+abs(pos.z)/7.);
			ambientLeft += samplee*(clamp(-pos.x,0.0,1.0)+clamp(pos.y/7.,0.0,1.0)+abs(pos.z)/7.);
			ambientRight += samplee*(clamp(pos.x,0.0,1.0)+clamp(pos.y/7.,0.0,1.0)+abs(pos.z)/7.);
			ambientB += samplee*(clamp(pos.z,0.0,1.0)+abs(pos.x)/7.+clamp(pos.y/7.,0.0,1.0));
			ambientF += samplee*(clamp(-pos.z,0.0,1.0)+abs(pos.x)/7.+clamp(pos.y/7.,0.0,1.0));
			ambientDown += samplee*(clamp(pos.y/6.,0.0,1.0)+abs(pos.x)/7.+abs(pos.z)/7.);

			/*
			ambientUp += samplee*(pos.y);
			ambientLeft += samplee*(clamp(-pos.x,0.0,1.0));
			ambientRight += samplee*(clamp(pos.x,0.0,1.0));
			ambientB += samplee*(clamp(pos.z,0.0,1.0));
			ambientF += samplee*(clamp(-pos.z,0.0,1.0));
			ambientDown += samplee*(clamp(pos.y/6.,0.0,1.0))*0;
			*/

	}


	vec2 planetSphere = vec2(0.0);
	vec3 sky = vec3(0.0);
	vec3 skyAbsorb = vec3(0.0);

	float sunVis = clamp(sunElevation,0.0,0.05)/0.05*clamp(sunElevation,0.0,0.05)/0.05;
	float moonVis = clamp(-sunElevation,0.0,0.05)/0.05*clamp(-sunElevation,0.0,0.05)/0.05;

	zenithColor = calculateAtmosphere(vec3(0.0), vec3(0.0,1.0,0.0), vec3(0.0,1.0,0.0), sunVec, -sunVec, planetSphere, skyAbsorb, 25,tempOffsets.x);
	skyAbsorb = vec3(0.0);
	vec3 absorb = vec3(0.0);
	sunColor = calculateAtmosphere(vec3(0.0), sunVec, vec3(0.0,1.0,0.0), sunVec, -sunVec, planetSphere, skyAbsorb, 25,0.0);
	sunColor = sunColorBase/4000. * skyAbsorb;

	skyAbsorb = vec3(0.0);
				   
	vec3 modSunVec = sunVec*0.96+vec3(0.0,0.04,1.0);
	vec3 modSunVec2 = sunVec*0.96+vec3(0.0,0.04,1.0);
	if (modSunVec2.y > modSunVec.y) modSunVec = modSunVec2;
	sunColorCloud = calculateAtmosphere(vec3(0.0), modSunVec, vec3(0.0,1.0,0.0), sunVec, -sunVec, planetSphere, skyAbsorb, 25,0.);
	sunColorCloud = sunColorBase/4000. * skyAbsorb ;

					   
	moonColor = calculateAtmosphere(vec3(0.0), -sunVec, vec3(0.0,1.0,0.0), sunVec, -sunVec, planetSphere, skyAbsorb, 25,0.5);
	moonColor = moonColorBase/4000.0*0.55;

	skyAbsorb = vec3(0.0);
	modSunVec = -sunVec*0.96+vec3(0.0,0.04,1.0);
	modSunVec2 = -sunVec*0.96+vec3(0.0,0.04,1.0);
	if (modSunVec2.y > modSunVec.y) modSunVec = modSunVec2;
	moonColorCloud = calculateAtmosphere(vec3(0.0), modSunVec, vec3(0.0,1.0,0.0), sunVec, -sunVec, planetSphere, skyAbsorb, 25,0.5);

	moonColorCloud = moonColorBase/4000.0*0.55;
	#ifndef CLOUDS_SHADOWS
	sunColor *= (1.0-rainStrength*vec3(0.96,0.95,0.94));
	moonColor *= (1.0-rainStrength*vec3(0.96,0.95,0.94));
	#endif
	lightSourceColor = sunVis >= 1e-5 ? sunColor * sunVis : moonColor * moonVis;

	float lightDir = float( sunVis >= 1e-5)*2.0-1.0;


	//Fake bounced sunlight
	vec3 bouncedSun = lightSourceColor/3.14/3.0*0.3*clamp(lightDir*sunVec.y,0.0,1.0)*clamp(lightDir*sunVec.y,0.0,1.0);
	vec3 cloudAmbientSun = (sunColorCloud)*0.006;
	vec3 cloudAmbientMoon = (moonColorCloud)*0.006;
	ambientUp += bouncedSun*clamp(-lightDir*sunVec.y+2.,0.,4.0) + cloudAmbientSun*clamp(sunVec.y+2.,0.,4.0) + cloudAmbientMoon*clamp(-sunVec.y+2.,0.,4.0);
	ambientLeft += bouncedSun*clamp(lightDir*sunVec.x+2.,0.0,4.) + cloudAmbientSun*clamp(-sunVec.x+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(sunVec.x+2.,0.0,4.)*0.7;
	ambientRight += bouncedSun*clamp(-lightDir*sunVec.x+2.,0.0,4.) + cloudAmbientSun*clamp(sunVec.x+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(-sunVec.x+2.,0.0,4.)*0.7;
	ambientB += bouncedSun*clamp(-lightDir*sunVec.z+2.,0.0,4.) + cloudAmbientSun*clamp(sunVec.z+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(-sunVec.z+2.,0.0,4.)*0.7;
	ambientF += bouncedSun*clamp(lightDir*sunVec.z+2.,0.0,4.) + cloudAmbientSun*clamp(-sunVec.z+2.,0.0,4.)*0.7 + cloudAmbientMoon*clamp(sunVec.z+2.,0.0,4.)*0.7;
	ambientDown += bouncedSun*clamp(lightDir*sunVec.y+2.,0.0,4.)*0.7 + cloudAmbientSun*clamp(-sunVec.y+2.,0.0,4.)*0.5 + cloudAmbientMoon*clamp(sunVec.y+2.,0.0,4.)*0.5;
	avgSky += bouncedSun*0.5;

	vec3 rainNightBoost = moonColorCloud*rainStrength*0.005;
	ambientUp += rainNightBoost;
	ambientLeft += rainNightBoost;
	ambientRight += rainNightBoost;
	ambientB += rainNightBoost;
	ambientF += rainNightBoost;
	ambientDown += rainNightBoost;
	avgSky += rainNightBoost;

	float avgLuma = 0.0;
	float m2 = 0.0;
	int n=100;
	vec2 clampedRes = max(1.0/texelSize,vec2(1920.0,1080.));
	float avgExp = 0.0;
	float avgB = 0.0;
	vec2 resScale = vec2(1920.,1080.)/clampedRes;
	const int maxITexp = 50;
	float w = 0.0;
	for (int i = 0; i < maxITexp; i++){
			vec2 ij = R2_samples((frameCounter%2000)*maxITexp+i);
			vec2 tc = 0.5 + (ij-0.5) * 0.7;
			vec3 sp = texture2D(colortex6,tc/16. * resScale+vec2(0.375*resScale.x+4.5*texelSize.x,.0)).rgb;
			avgExp += log(luma(sp));
			avgB += log(min(dot(sp,vec3(0.07,0.22,0.71)),8e-2));
	}

	avgExp = exp(avgExp/maxITexp);
	avgB = exp(avgB/maxITexp);

	avgBrightness = clamp(mix(avgExp,texelFetch2D(colortex4,ivec2(10,37),0).g,0.95),0.00003051757,65000.0);

	float L = max(avgBrightness,1e-8);
	float keyVal = 1.03-2.0/(log(L*0.6+1.0)/log(10.0)+2.0);
	float expFunc = 0.5+0.5*tanh(log(L));
	float targetExposure = 0.18/log2(L*2.0+1.04)*0.9;

	avgL2 = clamp(mix(avgB,texelFetch2D(colortex4,ivec2(10,37),0).b,0.975),0.00003051757,65000.0);
	float targetrodExposure = max(0.012/log2(avgL2+1.002),0.0);


	exposure=targetExposure*EXPOSURE_MULTIPLIER;


	rodExposure = targetrodExposure;

	#ifndef AUTO_EXPOSURE
	 exposure = Manual_exposure_value;
	 rodExposure = clamp(log(Manual_exposure_value*2.0+1.0)-0.1,0.0,2.0);
	#endif
	float modWT = (worldTime%24000)*1.0;

	float fogAmount0 = 1/3000.+FOG_TOD_MULTIPLIER*(1/180.*(clamp(modWT-11000.,0.,2000.0)/2000.+(1.0-clamp(modWT,0.,3000.0)/3000.))*(clamp(modWT-11000.,0.,2000.0)/2000.+(1.0-clamp(modWT,0.,3000.0)/3000.)) + 1/200.*clamp(modWT-13000.,0.,1000.0)/1000.*(1.0-clamp(modWT-23000.,0.,1000.0)/1000.));
	VFAmount = CLOUDY_FOG_AMOUNT*(fogAmount0*fogAmount0+FOG_RAIN_MULTIPLIER*1.8/20000.*rainStrength);
	fogAmount = BASE_FOG_AMOUNT*(fogAmount0+max(FOG_RAIN_MULTIPLIER*1/10.*rainStrength , FOG_TOD_MULTIPLIER*1/50.*clamp(modWT-13000.,0.,1000.0)/1000.*(1.0-clamp(modWT-23000.,0.,1000.0)/1000.)));
}
