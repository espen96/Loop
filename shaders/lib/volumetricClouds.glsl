

					   
				
					   
							  
	 
					   
float cloudSpeed = frameTimeCounter * CLOUDS_SPEED;							  
float cloud_height = 1500.;
float maxHeight = 3200.;


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
	return texture2D( tex, uv);
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
	vec2 xy = texture2D(noisetex, coord).yx;

	return mix(xy.r,xy.g, f.y);
}


//Cloud without 3D noise, is used to exit early lighting calculations if there is no cloud
float cloudCov(in vec3 pos,vec3 samplePos){
	float mult = max(pos.y-2000.0,0.0)/2000.0;
	float mult2 = max(-pos.y+2000.0,0.0)/500.0;
	float coverage = max((texture2D(noisetex,(samplePos.xz+sin(dot(samplePos.xz, vec2(0.5))/1000.)*600)/15000.).r+0.9*rainStrength+0.1)/(1.1+0.9*rainStrength) + cloudCoverage/2.5,0.0);
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

float cloudVol2(in vec3 pos,in vec3 samplePos,in float cov, in int LoD){
	float noise = 0.0;
	float totalWeights = 0.0;
	float pw = log(fbmPower1);
	float pw2 = log(fbmPower2);
	for (int i = 0; i <= LoD; i++){
	  float	weight = exp(-i*pw2);
		noise += weight - densityAtPos(samplePos*exp(i*pw))*weight;
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
		vec3 samplePos = pos*vec3(1.0,1./32.,1.0)/4+cloudSpeed*vec3(0.5,0.,0.5)*25.;
		float coverage = (texture2D(noisetex,(samplePos.xz+sin(dot(samplePos.xz, vec2(0.5))/1000.)*600)/15000.).r+0.9*rainStrength+0.1)/(1.1+0.9*rainStrength)-0.1;
		float cloud = coverage*coverage*3.0 - mult*mult*mult*3.0 - mult2*mult2*0.9;
		return max(cloud, 0.0);
	}


float getCloudDensity(in vec3 pos, in int LoD){
	vec3 samplePos = pos*vec3(1.0,1./32.,1.0)/4 + cloudSpeed*vec3(0.5,0.,0.5)*25.;
	float coverageSP = cloudCov(pos,samplePos);
	if (coverageSP > 0.001) {
		if (LoD < 0)
			return max(coverageSP - 0.27*(fbmAmount+rainStrength),0.0);
		return cloudVol(pos,samplePos,coverageSP, LoD);
	}
	else
		return 0.0;
}



float getCloudDensity2(in vec3 pos, in int LoD){
	vec3 samplePos = pos*vec3(1.0,1./32.,1.0)/4 + cloudSpeed*vec3(0.5,0.,0.5)*25.;
	float coverageSP = cloudCov(pos,samplePos);
	if (coverageSP > 0.001) {
		if (LoD < 0)
			return max(coverageSP - 0.27*(fbmAmount+rainStrength),0.0);
		return cloudVol2(pos,samplePos,coverageSP, LoD);
	}
	else
		return 0.0;
}


//Mie phase function
float phaseg(float x, float g){
    float gg = g * g;
    return (gg * -0.25 /3.14 + 0.25 /3.14) * pow(-2.0 * (g * x) + (gg + 1.0), -1.5);
}

float calcShadow(vec3 pos, vec3 ray){
	float shadowStep = length(ray);
	float d = 0.0;
	for (int j=1;j<6;j++){
		float cloudS=1.0;
		d += cloudS*cloudDensity;

		}
	return max(exp(-shadowStep*d),exp(-0.25*shadowStep*d)*0.7);
}
float cirrusClouds(vec3 pos){
	vec2 pos2D = pos.xz/50000.0 + cloudSpeed/200.;
	float cirrusMap = clamp(texture2D(noisetex,pos2D.yx/6. ).b-0.7+0.7*rainStrength,0.0,1.0);
	float cloud = texture2D(noisetex, pos2D).r;
	float weights = 1.0;
	vec2 posMult = vec2(2.0,1.5);
	for (int i = 1; i < 4; i++){
		pos2D *= posMult;
		float weight =exp2(-i*1.0);
		cloud += texture2D(noisetex, pos2D).r*weight;

		weights += weight;
	}
	cloud = clamp(cloud*cirrusMap*0.25,0.0,1.0);
	return cloud/weights * float(abs(pos.y - 5500.0) < 200.0);
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


		float distW = length(worldV);
		worldV = normalize(worldV)*300000. + cameraPosition; //makes max cloud distance not dependant of render distance
		dV_view = normalize(dV_view);

		//setup ray to start at the start of the cloud plane and end at the end of the cloud plane
		dV_view *= max(maxHeight-cloud_height, 0.0)/dV_view.y/maxIT_clouds;
		vec3 startOffset = dV_view*dither;

		progress_view = startOffset + cameraPosition + dV_view*(cloud_height-cameraPosition.y)/(dV_view.y);


		if (worldV.y < cloud_height) return vec4(0.,0.,0.,1.);	//don't trace if no intersection is possible



		float shadowStep = 200.;
		vec3 dV_Sun = normalize(mat3(gbufferModelViewInverse)*sunVec)*shadowStep;

		float mult = length(dV_view);


		color = vec3(0.0);

		total_extinction = 1.0;
		float SdotV = dot(sunVec,normalize(fragpositi));
		float SdotV01 = SdotV*0.5+0.5;
		//fake multiple scattering approx 1 (from horizon zero down clouds)

		float mieDay = max(phaseg(SdotV, cloudMieG),phaseg(SdotV, cloudMieG2)*cloudMie2Multiplier);
		float mieNight = max(phaseg(-SdotV, cloudMieG),phaseg(-SdotV, cloudMieG2)*cloudMie2Multiplier);
		
		
		
		vec3 sunContribution = mieDay*sunColor*2.14;
		vec3 moonContribution = mieNight*moonColor*10.14;
		vec3 skyCol0 = avgAmbient*(1.0-rainStrength*0.8);

		float powderMulMoon = 1.0;
		float powderMulSun = 1.0;

		for (int i=0;i<maxIT_clouds;i++) {
			float cloud = getCloudDensity(progress_view, cloudLoD);
				if (cloud > 0.0001){
					float muS = cloud*cloudDensity;
					float muE =	cloud*cloudDensity;
					float muEshD = 0.0;
					if (sunContribution.g > 1e-5){
					#ifndef TOASTER
						for (int j=1;j<10;j++){
					#else	
						for (int j=1;j<8;j++){
					#endif	
						
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
						for (int j=1;j<8;j++){
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
					float sunShadow = max(exp(-muEshD),0.7*exp(-0.25*muEshD))*mix(1.0, powder,  h);
					float moonShadow = max(exp2(-muEshN),0.7*exp(-0.25*muEshN))*mix(1.0, powder,  h);
					float ambientPowder = mix(1.0,powder,h);
					vec3 S = vec3(sunContribution*sunShadow+moonShadow*moonContribution+skyCol0*ambientPowder);


					vec3 Sint=(S - S * exp(-mult*muE)) / (muE);
					color += muS*Sint*total_extinction;
					total_extinction *= exp(-muE*mult);

					if (total_extinction < 1e-5) break;
				}
				progress_view += dV_view;
			}



		float cosY = normalize(dV_view).y;


		return mix(vec4(color,clamp(total_extinction,0.0,1.0)),vec4(0.0,0.0,0.0,1.0),1-smoothstep(0.02,0.15,cosY));

}
