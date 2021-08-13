
#define VOLUMETRIC_CLOUDS

#define cloud_LevelOfDetail 2		// Number of fbm noise iterations for on-screen clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define cloud_ShadowLevelOfDetail -1	// Number of fbm noise iterations for the shadowing of on-screen clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define cloud_LevelOfDetailLQ -1	// Number of fbm noise iterations for reflected clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define cloud_ShadowLevelOfDetailLQ -1	// Number of fbm noise iterations for the shadowing of reflected clouds (-1 is no fbm)	[-1 0 1 2 3 4 5 6 7 8]
#define minRayMarchSteps 25		// Number of ray march steps towards zenith for on-screen clouds	[20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200]
#define maxRayMarchSteps 80		// Number of ray march steps towards horizon for on-screen clouds	[20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200]
#define minRayMarchStepsLQ 5	// Number of ray march steps towards zenith for reflected clouds	[5  10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90 95 100]
#define maxRayMarchStepsLQ 20		// Number of ray march steps towards horizon for reflected clouds	[  5  10  15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90 95 100]




#define VOLUMETRIC_CLOUDS

#define CLOUDS_SPEED 1 //[0.0 0.25 0.4 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.1 1.2 1.3 1.4 1.5 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
			   
#define cloudMieG 0.55 // Values close to 1 will create a strong peak of luminance around the sun and weak elsewhere, values close to 0 means uniform fog. [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]
#define cloudMieG2 0.2 // Multiple scattering approximation. Values close to 1 will create a strong peak of luminance around the sun and weak elsewhere, values close to 0 means uniform fog. [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]
#define cloudMie2Multiplier 0.7 // Multiplier for multiple scattering approximation  [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 ]



#define cloudDensity 0.015		// Cloud Density, 0.02-0.04 is around irl values	[0.0010 0.0011 0.0013 0.0015 0.0017 0.0020 0.0023 0.0026 0.0030 0.0034 0.0039 0.0045 0.0051 0.0058 0.0067 0.0077 0.0088 0.0101 0.0115 0.0132 0.0151 0.0173 0.0199 0.0228 0.0261 0.0299 0.0342 0.0392 0.0449 0.0514 0.0589 0.0675 0.0773 0.0885 0.1014 0.1162 0.1331 0.1524 0.1746 0.2000]
#define cloudCoverage -0.15			// Cloud coverage	[-1.00 -0.98 -0.96 -0.94 -0.92 -0.90 -0.88 -0.86 -0.84 -0.82 -0.80 -0.78 -0.76 -0.74 -0.72 -0.70 -0.68 -0.66 -0.64 -0.62 -0.60 -0.58 -0.56 -0.54 -0.52 -0.50 -0.48 -0.46 -0.44 -0.42 -0.40 -0.38 -0.36 -0.34 -0.32 -0.30 -0.28 -0.26 -0.24 -0.22 -0.20 -0.18 -0.16 -0.14 -0.12 -0.10 -0.08 -0.06 -0.04 -0.02 0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44 0.46 0.48 0.50 0.52 0.54 0.56 0.58 0.60 0.62 0.64 0.66 0.68 0.70 0.72 0.74 0.76 0.78 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98 1.00]
#define fbmAmount 1.00 		// Amount of noise added to the cloud shape	[0.00 0.02 0.04 0.06 0.08 0.10 0.12 0.14 0.16 0.18 0.20 0.22 0.24 0.26 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0.44 0.46 0.48 0.50 0.52 0.54 0.56 0.58 0.60 0.62 0.64 0.66 0.68 0.70 0.72 0.74 0.76 0.78 0.80 0.82 0.84 0.86 0.88 0.90 0.92 0.94 0.96 0.98 1.00 1.02 1.04 1.06 1.08 1.10 1.12 1.14 1.16 1.18 1.20 1.22 1.24 1.26 1.28 1.30 1.32 1.34 1.36 1.38 1.40 1.42 1.44 1.46 1.48 1.50 1.52 1.54 1.56 1.58 1.60 1.62 1.64 1.66 1.68 1.70 1.72 1.74 1.76 1.78 1.80 1.82 1.84 1.86 1.88 1.90 1.92 1.94 1.96 1.98 2.00 2.02 2.04 2.06 2.08 2.10 2.12 2.14 2.16 2.18 2.20 2.22 2.24 2.26 2.28 2.30 2.32 2.34 2.36 2.38 2.40 2.42 2.44 2.46 2.48 2.50 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 2.68 2.70 2.72 2.74 2.76 2.78 2.80 2.82 2.84 2.86 2.88 2.90 2.92 2.94 2.96 2.98 3.00]
#define fbmPower1 2.90	// Higher values increases high frequency details of the cloud shape	[1.50 1.52 1.54 1.56 1.58 1.60 1.62 1.64 1.66 1.68 1.70 1.72 1.74 1.76 1.78 1.80 1.82 1.84 1.86 1.88 1.90 1.92 1.94 1.96 1.98 2.00 2.02 2.04 2.06 2.08 2.10 2.12 2.14 2.16 2.18 2.20 2.22 2.24 2.26 2.28 2.30 2.32 2.34 2.36 2.38 2.40 2.42 2.44 2.46 2.48 2.50 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 2.68 2.70 2.72 2.74 2.76 2.78 2.80 2.82 2.84 2.86 2.88 2.90 2.92 2.94 2.96 2.98 3.00 3.02 3.04 3.06 3.08 3.10 3.12 3.14 3.16 3.18 3.20 3.22 3.24 3.26 3.28 3.30 3.32 3.34 3.36 3.38 3.40 3.42 3.44 3.46 3.48 3.50 3.52 3.54 3.56 3.58 3.60 3.62 3.64 3.66 3.68 3.70 3.72 3.74 3.76 3.78 3.80 3.82 3.84 3.86 3.88 3.90 3.92 3.94 3.96 3.98 4.00]
#define fbmPower2 3.00	// Lower values increases high frequency details of the cloud shape	[1.50 1.52 1.54 1.56 1.58 1.60 1.62 1.64 1.66 1.68 1.70 1.72 1.74 1.76 1.78 1.80 1.82 1.84 1.86 1.88 1.90 1.92 1.94 1.96 1.98 2.00 2.02 2.04 2.06 2.08 2.10 2.12 2.14 2.16 2.18 2.20 2.22 2.24 2.26 2.28 2.30 2.32 2.34 2.36 2.38 2.40 2.42 2.44 2.46 2.48 2.50 2.52 2.54 2.56 2.58 2.60 2.62 2.64 2.66 2.68 2.70 2.72 2.74 2.76 2.78 2.80 2.82 2.84 2.86 2.88 2.90 2.92 2.94 2.96 2.98 3.00 3.02 3.04 3.06 3.08 3.10 3.12 3.14 3.16 3.18 3.20 3.22 3.24 3.26 3.28 3.30 3.32 3.34 3.36 3.38 3.40 3.42 3.44 3.46 3.48 3.50 3.52 3.54 3.56 3.58 3.60 3.62 3.64 3.66 3.68 3.70 3.72 3.74 3.76 3.78 3.80 3.82 3.84 3.86 3.88 3.90 3.92 3.94 3.96 3.98 4.00]


float cloudSpeed  = (frameTimeCounter);	


vec2 windGenerator(float t) {
    float tx = t;

    vec2 p = vec2(sin(2.2 * tx)+ - cos(1.4 * tx), cos(1.3 * t) + sin(-1.9 * t));
    p.y *= 0.0015;
    p.x *= 0.001;
 	return p;
}

   
							  
vec2 wind = windGenerator(frameTimeCounter*0.001)*20000*CLOUDS_SPEED; 	 				   
vec3 cloudSpeed2 =  vec3(frameTimeCounter*wind.x,0.0,frameTimeCounter*wind.y)*0.5;

						  
float cloud_height = 1500.0;
float maxHeight = 3200.0;


#ifdef HQ_CLOUDS
int maxIT_clouds = minRayMarchSteps;
int maxIT = maxRayMarchSteps;
#else
int maxIT_clouds = minRayMarchStepsLQ;
int maxIT = maxRayMarchStepsLQ;
#endif 



#ifdef HQ_CLOUDS
const int cloudLoD = cloud_LevelOfDetail;
const int cloudShadowLoD = cloud_ShadowLevelOfDetail;
#else
const int cloudLoD = cloud_LevelOfDetailLQ;
const int cloudShadowLoD = cloud_ShadowLevelOfDetailLQ;
#endif
vec4 smoothfilter(in sampler2D tex, in vec2 uv)
{
	uv = uv*512.0 + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + (fuv*fuv)*(3.0-2.0*fuv);
	uv = uv/512.0 - 0.5/512.0;
	return texture
( tex, uv);
}



//3D noise from 2d texture
float densityAtPos(in vec3 pos)
{

	pos /= 18.;
	pos.xz *= 0.5;


	vec3 p = floor(pos);
	vec3 f = fract(pos);

	f = (f*f) * (3.-2.*f);

	vec2 uv =  p.xz + f.xz + p.y * vec2(0.0,193.0);

	vec2 coord =  uv / 512.0;
	//The y channel has an offset to avoid using two textures fetches
	vec2 xy = texture
(noisetex, coord).yx;

	return mix(xy.r,xy.g, f.y);
}


//Cloud without 3D noise, is used to exit early lighting calculations if there is no cloud
float cloudCov(in vec3 pos,vec3 samplePos){
	float mult = max(pos.y-2000.0,0.0)/2000.0;
	float mult2 = max(-pos.y+2000.0,0.0)/500.0;
	float coverage = max((texture
(noisetex,(samplePos.xz + (wind*100) +sin(dot(samplePos.xz, vec2(0.5))/1000.)*600)/15000.).r+0.9*rainStrength+0.1)/(1.1+0.9*rainStrength) + ((sin(0.2 * frameTimeCounter*0.05)+ - cos(0.4 * frameTimeCounter*0.05))*cloudCoverage)/2.5,0.0);
	float cloud = coverage*coverage*3.0 - mult*mult*mult*3.0 - mult2*mult2*0.9;
	return max(cloud, 0.0);
}
//Erode cloud with 3d Perlin-worley noise, actual cloud value
float cloudVol(in vec3 pos,in vec3 samplePos,in float cov, in int LoD){
	float noise = 0.0;
	float totalWeights = 0.0;
	float pw = log(fbmPower1);
	float pw2 = log(fbmPower2);
	for (int i = 0; i <= LoD; i++){
	  float	weight = exp(-i*pw2);
		noise += weight - densityAtPos(samplePos*8.*exp(i*pw))*weight;
		totalWeights += weight;
	}
	noise /= totalWeights;
	noise = noise*noise;
	float cloud = max(cov-noise*noise*(1.1+rainStrength)*fbmAmount,0.0);
	//float cloud = clamp(cov-0.1*(0.2+mult2),0.0,1.0);
	return cloud;
}





	//Low quality cloud, noise is replaced by the average noise value, used for shadowing
	float cloudVolLQ(in vec3 pos){
		float mult = max(pos.y-2000.0,0.0)/2000.0;
		float mult2 = max(-pos.y+2000.0,0.0)/500.0;
		float mult3 = (pos.y-1500)/2500.0+rainStrength*0.4;
		vec3 samplePos = pos*vec3(1.0,1./32.,1.0)/4+cloudSpeed2;
		float coverage = (texture
(noisetex,(samplePos.xz+sin(dot(samplePos.xz, vec2(0.5))/1000.)*600)/15000.).r+0.9*rainStrength+0.1)/(1.1+0.9*rainStrength)-0.1;
		float cloud = coverage*coverage*3.0 - mult*mult*mult*3.0 - mult2*mult2*0.9;
		return max(cloud, 0.0);
	}


float getCloudDensity(in vec3 pos, in int LoD){
	vec3 samplePos = pos*vec3(1.0,1.0/32.0,1.0)/4 + cloudSpeed2;
	float coverageSP = cloudCov(pos,samplePos);
	if (coverageSP > 0.001) {
		if (LoD < 0)
			return max(coverageSP - 0.27*(fbmAmount+rainStrength),0.0);
		return cloudVol(pos,samplePos,coverageSP, LoD);
	}
	else
		return 0.0;
}


//Mie phase function
float phaseg(float x, float g){
    float gg = g * g;
    return (gg * -0.25 /3.14 + 0.25 /3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}


vec4 renderClouds(vec3 fragpositi, vec3 color,float dither,vec3 sunColor,vec3 moonColor,vec3 avgAmbient) {


		#ifndef VOLUMETRIC_CLOUDS
			return vec4(0.0,0.0,0.0,1.0);
		#endif
		//setup ray in projected shadow map space
		bool land = false;

		float SdotU = dot(normalize(fragpositi.xyz),sunVec);
		float z2 = length(fragpositi);
		float z = -fragpositi.z;


		//project pixel position into projected shadowmap space
		vec4 fragposition = gbufferModelViewInverse*vec4(fragpositi,1.0);

		vec3 worldV = normalize(fragposition.rgb);
		float VdotU = worldV.y;
		maxIT_clouds = int(clamp(maxIT_clouds/sqrt(VdotU),0.0,maxIT*1.0));
		//worldV.y -= -length(worldV.xz)/sqrt(-length(worldV.xz/earthRad)*length(worldV.xz/earthRad)+earthRad);

		//project view origin into projected shadowmap space
		vec4 start = (gbufferModelViewInverse*vec4(0.0,0.0,0.,1.));
		vec3 dV_view = worldV;


		vec3 progress_view = dV_view*dither+cameraPosition;

		float vL = 0.0;
		float total_extinction = 1.0;


//		float distW = length(worldV);
		worldV = normalize(worldV)*100000. + cameraPosition; //makes max cloud distance not dependant of render distance
		dV_view = normalize(dV_view);

		//setup ray to start at the start of the cloud plane and end at the end of the cloud plane
		dV_view *= max(maxHeight-cloud_height, 0.0)/dV_view.y/maxIT_clouds;
		vec3 startOffset = dV_view*dither;

		progress_view = startOffset + cameraPosition + dV_view*(cloud_height-cameraPosition.y)/(dV_view.y);


		if (worldV.y < cloud_height) return vec4(0.,0.,0.,1.);	//don't trace if no intersection is possible



		float shadowStep = 200.;
		vec3 dV_Sun = normalize(mat3(gbufferModelViewInverse)*sunVec)*shadowStep;

		float mult = length(dV_view);


//		color = vec3(0.0);

		total_extinction = 1.0;
		float SdotV = dot(sunVec,normalize(fragpositi));
		float SdotV01 = SdotV*0.5+0.5;


		//fake multiple scattering approx 1 (from horizon zero down clouds)

		float mieDay = max(phaseg(SdotV, cloudMieG),phaseg(SdotV, cloudMieG2)*cloudMie2Multiplier);
		float mieDayMulti = phaseg(SdotV, 0.2);										 
		float mieNight = max(phaseg(-SdotV, cloudMieG),phaseg(-SdotV, cloudMieG2)*cloudMie2Multiplier);
		float mieNightMulti = phaseg(-SdotV, 0.2);											
		
		
		
		vec3 sunContribution = mieDay*sunColor*2.14;
		vec3 sunContributionMulti = mieDayMulti*sunColor;												   
		vec3 moonContribution = mieNight*moonColor*10.14;
		vec3 moonContributionMulti = mieNightMulti*moonColor;													   
		float ambientMult = exp(-(cloudCoverage+0.24+0.8*rainStrength)*cloudDensity*75.0);
		vec3 skyCol0 = avgAmbient * ambientMult;

		float powderMulMoon = 1.0;
		float powderMulSun = 1.0;

		for (int i=0;i<maxIT_clouds;i++) {
			float cloud = getCloudDensity(progress_view, cloudLoD);
				if (cloud > 0.0001){
					float muS = cloud*cloudDensity;
					float muE =	cloud*cloudDensity;
					float muEshD = 0.0;
					if (sunContribution.g > 1e-5){
	
						for (int j=1;j<6;j++){
				
						
							vec3 shadowSamplePos = progress_view+dV_Sun*j;
							if (shadowSamplePos.y < maxHeight)
							{
								float cloudS=getCloudDensity(vec3(shadowSamplePos), cloudShadowLoD);
								muEshD += cloudS*cloudDensity*shadowStep;
							}
						}
					}
					float muEshN = 0.0;
					if (moonContribution.g > 1e-5){
						for (int j=1;j<6;j++){
							vec3 shadowSamplePos = progress_view-dV_Sun*j;
							if (shadowSamplePos.y < maxHeight)
							{
								float cloudS=getCloudDensity(vec3(shadowSamplePos), cloudShadowLoD);
								muEshN += cloudS*cloudDensity*shadowStep;
							}
						}
					}
					//fake multiple scattering approx 2  (from horizon zero down clouds)
					float h = 0.5-0.5*clamp(progress_view.y/4000.-1500./4000.,0.0,1.0);
					float powder = 1.0-exp(-muE*mult);
//					float sunShadow = max(exp(-muEshD),0.1*exp(-0.25*muEshD))*mix(1.0, powder,  h);
//					float moonShadow = max(exp(-muEshN),0.1*exp(-0.25*muEshN))*mix(1.0, powder,  h);
					float sunShadow = exp(-muEshD);
					float moonShadow = exp(-muEshN);
					float moonShadowMulti = exp(-log(muEshN*0.15+0.5)) * powder;			
					float sunShadowMulti = exp(-log(muEshD*0.15+0.5)) * powder;					
					float ambientPowder = mix(1.0,powder,h * ambientMult);
					vec3 S = vec3(sunContribution*sunShadow+moonShadow*moonContribution+skyCol0*ambientPowder);
//					vec3 S = vec3(sunContribution*sunShadow + sunShadowMulti*sunContributionMulti + moonShadowMulti*moonContributionMulti +  moonShadow*moonContribution+skyCol0*ambientPowder);


					vec3 Sint=(S - S * exp(-mult*muE)) / (muE);
					color += muS*Sint*total_extinction;
					total_extinction *= exp(-muE*mult);

					if (total_extinction < 1e-5) break;
				}
				progress_view += dV_view;
			}



		vec3 normView = normalize(dV_view);
		// Assume fog color = sky gradient at long distance
		vec3 fogColor = skyFromTex(normView, colortex4)/150.;
		float dist = (cloud_height - cameraPosition.y)/normalize(dV_view).y;
		float fog = exp(-dist/20000.0*(1.0+rainStrength*2.));

		return mix(vec4(fogColor,0.0), vec4(color,clamp(total_extinction,0.0,1.0)), fog);

}
