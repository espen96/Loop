//#define VL_Clouds_Shadows
#define CLOUDS_SHADOWS_STRENGTH 1.0 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define VL_SAMPLES2 4 //[4 6 8 10 12 14 16 20 24 30 40 50]
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
float luma(vec3 color) {
	return dot(color,vec3(0.299, 0.587, 0.114));
}

float phaseRayleigh(float cosTheta) {
	const vec2 mul_add = vec2(0.1, 0.28) /acos(-1.0);
	return cosTheta * mul_add.x + mul_add.y; // optimized version from [Elek09], divided by 4 pi for energy conservation
}
float cloudVol2(in vec3 pos){

	vec3 samplePos = pos*vec3(1.0,1./16.,1.0)+frameTimeCounter*vec3(0.5,0.,0.5)*5.;
	float coverage = mix(exp2(-(pos.y-SEA_LEVEL)*(pos.y-SEA_LEVEL)/10000.),1.0,rainStrength*0.5);
	float noise = densityAtPos(samplePos*12.);
	float unifCov = exp2(-max(pos.y-SEA_LEVEL,0.0)/50.);

	float cloud = pow(clamp(coverage-noise-0.76,0.0,1.0),2.)*1200./0.23/(coverage+0.01)*VFAmount*600+unifCov*60.*fogAmount;

return cloud;
}

float cloudVol(in vec3 pos){

	vec3 samplePos = pos*vec3(1.0,1./16.,1.0)+frameTimeCounter*vec3(0.5,0.,0.5)*5.;
	float coverage = mix(exp2(-(pos.y-SEA_LEVEL)*(pos.y-SEA_LEVEL)/10000.),1.0,rainStrength*0.5);
	float noise = densityAtPos(samplePos*12.);
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
	dV *= maxLength;
	dVWorld *= maxLength;

	//apply dither
	vec3 progress = start.xyz;
	vec3 progressW = gbufferModelViewInverse[3].xyz+cameraPosition;
		vec3 vL = vec3(0.);

		float SdotV = dot(sunVec,normalize(fragpos))*lightCol.a;
		float dL = length(dVWorld);
		//Mie phase + somewhat simulates multiple scattering (Horizon zero down cloud approx)
		float mie = mix(phaseg(SdotV,fog_mieg1),phaseg(SdotV,fog_mieg2),0.5);
		float rayL = phaseRayleigh(SdotV);
	//	wpos.y = clamp(wpos.y,0.0,1.0);

		vec3 ambientCoefs = dVWorld/dot(abs(dVWorld),vec3(1.));

		vec3 ambientLight = ambientUp*clamp(ambientCoefs.y,0.,1.);
		ambientLight += ambientDown*clamp(-ambientCoefs.y,0.,1.);
		ambientLight += ambientRight*clamp(ambientCoefs.x,0.,1.);
		ambientLight += ambientLeft*clamp(-ambientCoefs.x,0.,1.);
		ambientLight += ambientB*clamp(ambientCoefs.z,0.,1.);
		ambientLight += ambientF*clamp(-ambientCoefs.z,0.,1.);

		vec3 skyCol0 = ambientLight*eyeBrightnessSmooth.y/vec3(240.)*Ambient_Mult*2.0;
		// Makes fog more white idk how to simulate it correctly
		vec3 sunColor = lightCol.rgb;
		skyCol0 = skyCol0.rgb;

		vec3 rC = vec3(fog_coefficientRayleighR*1e-6, fog_coefficientRayleighG*1e-5, fog_coefficientRayleighB*1e-5);
		vec3 mC = vec3(fog_coefficientMieR*1e-6, fog_coefficientMieG*1e-6, fog_coefficientMieB*1e-6);


		float mu = 1.0;
		float muS = 1.0*mu;
		vec3 absorbance = vec3(1.0);
		float expFactor = 11.0;
		vec3 WsunVec = mat3(gbufferModelViewInverse) * sunVec;
		for (int i=0;i<VL_SAMPLES2;i++) {
			float d = (pow(expFactor, float(i+dither)/float(VL_SAMPLES2))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
			float dd = pow(expFactor, float(i+dither)/float(VL_SAMPLES2)) * log(expFactor) / float(VL_SAMPLES2)/(expFactor-1.0);
			progress = start.xyz + d*dV;
			progressW = gbufferModelViewInverse[3].xyz+cameraPosition + d*dVWorld;
			//project into biased shadowmap space
			float distortFactor = calcDistort(progress.xy);
			vec3 pos = vec3(progress.xy*distortFactor, progress.z);
			float densityVol = cloudVol(progressW);
			float sh = 1.0;
			if (abs(pos.x) < 1.0-0.5/2048. && abs(pos.y) < 1.0-0.5/2048){
				pos = pos*vec3(0.5,0.5,0.5/6.0)+0.5;
				sh =  shadow2D( shadow, pos).x;
				#ifdef VL_Clouds_Shadows
				float cloudShadow = 0.0;
				const int rayMarchSteps = 6;
				for (int i = 0; i < rayMarchSteps; i++){
					vec3 cloudPos = progressW + WsunVec/abs(WsunVec.y)*(1500+(dither+i)/rayMarchSteps*1700-progressW.y);
					cloudShadow += getCloudDensity(cloudPos, 0);
				}
				cloudShadow = mix(1.0,exp(-cloudShadow*cloudDensity*1700/rayMarchSteps),mix(CLOUDS_SHADOWS_STRENGTH,1.0,rainStrength));
				sh *= cloudShadow;
				#endif
			}
			//Water droplets(fog)
			float density = densityVol*ATMOSPHERIC_DENSITY*mu*300.;
			//Just air
			vec2 airCoef = exp2(-max(progressW.y-SEA_LEVEL,0.0)/vec2(8.0e3, 1.2e3)*vec2(6.,7.0))*6.0;

			//Pbr for air, yolo mix between mie and rayleigh for water droplets
			vec3 rL = rC*airCoef.x;
			vec3 m = (airCoef.y+density)*mC;
			vec3 vL0 = sunColor*sh*(rayL*rL+m*mie) + skyCol0*(rL+m);
			vL += (vL0 - vL0 * exp(-(rL+m)*dd*dL)) / ((rL+m)+0.00000001)*absorbance;
			absorbance *= clamp(exp(-(rL+m)*dd*dL),0.0,1.0);
		}
	return mat2x3(vL,absorbance);



}
