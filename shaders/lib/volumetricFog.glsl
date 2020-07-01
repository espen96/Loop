

float luma(vec3 color) {
	return dot(color,vec3(0.299, 0.587, 0.114));
}

float phaseRayleigh(float cosTheta) {
	const vec2 mul_add = vec2(0.1, 0.28) /acos(-1.0);
	return cosTheta * mul_add.x + mul_add.y; // optimized version from [Elek09], divided by 4 pi for energy conservation
}


float cloudVol(in vec3 pos){
	float unifCov = exp2(-max(pos.y-SEA_LEVEL,0.0)/50.);
	float cloud = unifCov*60.*fogAmount;
  return cloud;
}
mat2x3 getVolumetricRays(float dither,vec3 fragpos) {

	//project pixel position into projected shadowmap space
	vec3 wpos = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;

	vec3 dVWorld = (wpos-gbufferModelViewInverse[3].xyz);

	float maxLength = min(length(dVWorld),far)/length(dVWorld);

	dVWorld *= maxLength;


	vec3 progressW = gbufferModelViewInverse[3].xyz+cameraPosition;
		vec3 vL = vec3(0.);

		float SdotV = dot(sunVec,normalize(fragpos))*lightCol.a;
		float dL = length(dVWorld);
		//Mie phase + somewhat simulates multiple scattering (Horizon zero down cloud approx)
		float mie = max(phaseg(SdotV,fog_mieg1),1.0/13.0);
		float rayL = phaseRayleigh(SdotV);
	//	wpos.y = clamp(wpos.y,0.0,1.0);

		vec3 ambientCoefs = dVWorld/dot(abs(dVWorld),vec3(1.));

		vec3 ambientLight = ambientUp*clamp(ambientCoefs.y,0.,1.);
		ambientLight += ambientDown*clamp(-ambientCoefs.y,0.,1.);
		ambientLight += ambientRight*clamp(ambientCoefs.x,0.,1.);
		ambientLight += ambientLeft*clamp(-ambientCoefs.x,0.,1.);
		ambientLight += ambientB*clamp(ambientCoefs.z,0.,1.);
		ambientLight += ambientF*clamp(-ambientCoefs.z,0.,1.);

	vec3 skyCol0 = ambientLight*2.*eyeBrightnessSmooth.y/vec3(240.)*Ambient_Mult*2.0/PI;
														  
	vec3 sunColor = lightCol.rgb;

		vec3 rC = vec3(fog_coefficientRayleighR*1e-6, fog_coefficientRayleighG*1e-5, fog_coefficientRayleighB*1e-5);
		vec3 mC = vec3(fog_coefficientMieR*1e-6, fog_coefficientMieG*1e-6, fog_coefficientMieB*1e-6);


		float mu = 1.0;
		float muS = 1.0*mu;
		vec3 absorbance = vec3(1.0);
		float expFactor = 11.0;
		for (int i=0;i<VL_SAMPLES2;i++) {
			float d = (pow(expFactor, float(i+dither)/float(VL_SAMPLES2))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
			float dd = pow(expFactor, float(i+dither)/float(VL_SAMPLES2)) * log(expFactor) / float(VL_SAMPLES2)/(expFactor-1.0);
			progressW = gbufferModelViewInverse[3].xyz+cameraPosition + d*dVWorld;

    float density = cloudVol(progressW)*1.5*ATMOSPHERIC_DENSITY*mu*500.;
			//Just air
			vec2 airCoef = exp2(-max(progressW.y-SEA_LEVEL,0.0)/vec2(8.0e3, 1.2e3)*vec2(6.,7.0))*6.0;

			//Pbr for air, yolo mix between mie and rayleigh for water droplets
			vec3 rL = rC*(airCoef.x+density*0.15);
			vec3 m = (airCoef.y+density*1.85)*mC;
			vec3 vL0 = sunColor*(rayL*rL+m*mie)*0.75 + skyCol0*(rL+m);
			vL += vL0 * dd * dL *  absorbance;
			absorbance *= exp(-(rL+m)*dL*dd);
		}
	return mat2x3(vL,absorbance);



}
