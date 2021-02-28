






vec3 RT(vec3 dir,vec3 position,float noise, vec3 N,float transparent, vec2 lightmap, bool emissive, bool hand,float z){



	float depthmask = ld(z);
	float stepSize = STEP_LENGTH;
//	int maxSteps =   clamp(int(  (clamp((lightmap.x),0,1)*99)),6,99);
	int maxSteps = STEPS;

//	if (emissive) return vec3(1.1);
	if (hand) return vec3(1.1);



	bool istranparent = transparent > 0.0;
	vec3 clipPosition = toClipSpace3(position);
	
	float rayLength = ((position.z + dir.z * sqrt(3.0)*far) > -sqrt(3.0)*near) ?
	   								(-sqrt(3.0)*near -position.z) / dir.z : sqrt(3.0)*far;


	vec3 end = toClipSpace3(position+dir*rayLength);
	vec3 direction = end-clipPosition;  //convert to clip space

	float len = max(abs(direction.x)/texelSize.x,abs(direction.y)/texelSize.y)/stepSize;

	//get at which length the ray intersects with the edge of the screen
	vec3 maxLengths = (step(0.,direction)-clipPosition) / direction;
	float mult = min(min(maxLengths.x,maxLengths.y),maxLengths.z);
	vec3 stepv = direction/len;


	int iterations = min(int(min(len, mult*len)-2), maxSteps);

			
	
	
	//Do one iteration for closest texel (good contact shadows)
	vec3 spos = clipPosition*vec3(RENDER_SCALE,1.0) + stepv/stepSize*6.0;

	spos.xy += TAA_Offset*texelSize*0.5*RENDER_SCALE;

	float sp = sqrt(texelFetch2D(colortex4,ivec2(spos.xy/texelSize/4),0).w/65000.0);
	float currZ = linZ(spos.z);

	
	if( sp < currZ) {
		float dist = abs(sp-currZ)/currZ;
		
		if (dist <= 0.035) return vec3(spos.xy, invLinZ(sp))/vec3(RENDER_SCALE,1.0);

		
	}

	stepv *= vec3(RENDER_SCALE,1.0);

		
	spos += stepv*noise;

  for(int i = 0; i < iterations; i++){
      if (spos.x < 0.0 && spos.y < 0.0 && spos.z < 0.0 && spos.x > 1.0 && spos.y > 1.0 && spos.z > 1.0)
      return vec3(1.1);
  			
		// decode depth buffer
		float sp = sqrt(texelFetch2D(colortex4,ivec2(spos.xy/texelSize/4),0).w/65000.0);
			
		float currZ = linZ(spos.z);
			
		if( sp < currZ && abs(sp-ld(spos.z))/ld(spos.z) < 0.1) {
	

		
			float dist = abs(sp-currZ)/currZ;
			if (dist <= 0.035) return vec3(spos.xy, invLinZ(sp))/vec3(RENDER_SCALE,1.0);
		}
		

			spos += stepv;	

	}
	return vec3(1.1);

	
}



vec3 cosineHemisphereSample(vec2 Xi)
{
    float r = sqrt(Xi.x);
    float theta = 2.0 * 3.14159265359 * Xi.y;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec3(x, y, sqrt(clamp(1.0 - Xi.x,0.,1.)));
}
vec3 TangentToWorld(vec3 N, vec3 H)
{
    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(UpVector, N));
    vec3 B = cross(N, T);

    return vec3((T * H.x) + (B * H.y) + (N * H.z));
}
vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
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

//approximation from SMAA presentation from siggraph 2016
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

const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
							vec2(-1.,3.)/8.0,
							vec2(5.0,1.)/8.0,
							vec2(-3,-5.)/8.0,
							vec2(-5.,5.)/8.0,
							vec2(-7.,-1.)/8.,
							vec2(3,7.)/8.0,
							vec2(7.,-7.)/8.);
							
vec3 tonemap(vec3 col){
	return col/(1+luma(col));
}
vec3 invTonemap(vec3 col){
	return col/(1-luma(col));
}		



//////////////////////////////FANCY STUFF//////////////////////////////








float reconstructCSZ(float d, vec3 clipInfo) {
    return clipInfo[0] / (clipInfo[1] * d + clipInfo[2]);
}



vec3 reconstructCSPosition(vec2 S, float z, vec4 projInfo) {
    return vec3((S.xy * projInfo.xy + projInfo.zw) * z, z);
}



vec3 reconstructCSPositionFromDepth(vec2 S, float depth, vec4 projInfo, vec3 clipInfo) {
    return reconstructCSPosition(S, reconstructCSZ(depth, clipInfo), projInfo);
}


/** Helper for the common idiom of getting world-space position P.xyz from screen-space S = (x, y) in
    pixels and hyperbolic depth. 
    */
vec3 reconstructWSPositionFromDepth(vec2 S, float depth, vec4 projInfo, vec3 clipInfo, mat4x3 cameraToWorld) {
    return cameraToWorld * vec4(reconstructCSPositionFromDepth(S, depth, projInfo, clipInfo), 1.0);
}




/** Requires previousBuffer and previousDepthBuffer to be the same size as the output buffer

    Returns the reverse-reprojected value, and sets distance 
    to the WS distance from the expected WS reverse-reprojected position and the actual value.
*/




vec4 reverseReprojection(vec2 currentScreenCoord,  vec3 currentWSPosition, vec2 ssVelocity, 
		sampler2D previousBuffer, sampler2D previousDepthBuffer, vec2 inverseBufferSize, vec3 clipInfo, 
										vec4 projInfo, mat4x3 previousCameraToWorld, out float distance) {
			
			
    vec2 previousCoord  = currentScreenCoord - ssVelocity;
    vec2 normalizedPreviousCoord = previousCoord * inverseBufferSize;
    vec4 previousVal = texture2D(previousBuffer, normalizedPreviousCoord);

    float previousDepth = texture2D(previousDepthBuffer, normalizedPreviousCoord).r;
    vec3 wsPositionPrev = reconstructWSPositionFromDepth(previousCoord, previousDepth, projInfo, clipInfo, previousCameraToWorld);
    distance = length(currentWSPosition - wsPositionPrev);

    return previousVal;
}

/** Requires all input buffers to be the same size as the output buffer

    Returns the reverse-reprojected value from the closest layer, and sets distance 
    to the WS distance from the expected WS reverse-reprojected position and the actual value .
*/


vec4 twoLayerReverseReprojection(vec2 currentScreenCoord,  vec3 currentWSPosition, vec2 ssVelocity, 
	sampler2D previousBuffer, sampler2D previousDepthBuffer, sampler2D peeledPreviousBuffer, sampler2D peeledPreviousDepthBuffer, 
							vec2 inverseBufferSize, vec3 clipInfo, vec4 projInfo, mat4x3 previousCameraToWorld, out float distance) {

    vec2 previousCoord  = currentScreenCoord - ssVelocity;
    vec2 normalizedPreviousCoord = previousCoord * inverseBufferSize;
    vec4 previousVal = texture(previousBuffer, normalizedPreviousCoord);
    
    float previousDepth = texture(previousDepthBuffer, normalizedPreviousCoord).r;
    vec3 wsPositionPrev = previousCameraToWorld * vec4(reconstructCSPosition(previousCoord, reconstructCSZ(previousDepth, clipInfo), projInfo), 1.0);
    distance = length(currentWSPosition - wsPositionPrev);

    float previousPeeledDepth = texture(peeledPreviousDepthBuffer, normalizedPreviousCoord).r;
    vec3 wsPositionPeeledPrev = reconstructWSPositionFromDepth(previousCoord, previousPeeledDepth, projInfo, clipInfo, previousCameraToWorld);
    float distPeeled = length(currentWSPosition - wsPositionPeeledPrev);

    if (distPeeled < distance) {
        distance = distPeeled;
        previousVal = texture(peeledPreviousBuffer, normalizedPreviousCoord);
    }

    return previousVal;

}


/** Requires all input buffers to be the same size as the output buffer

    Returns the reverse-reprojected value from the closest layer, and sets distance 
    to the WS distance from the expected WS reverse-reprojected position and the actual value.
*/
vec4 reverseReprojection(vec2 currentScreenCoord,  sampler2D depthBuffer, 
                         sampler2D ssVelocityBuffer, vec2 ssVReadMultiplyFirst, vec2 ssVReadAddSecond, 
                         sampler2D previousBuffer, sampler2D previousDepthBuffer, vec2 inverseBufferSize,
                         vec3 clipInfo, vec4 projInfo, 
                         mat4x3 cameraToWorld, mat4x3 previousCameraToWorld, out float distance) {
    ivec2 C = ivec2(currentScreenCoord);
    vec2 ssV = texelFetch(ssVelocityBuffer, C, 0).rg * ssVReadMultiplyFirst + ssVReadAddSecond;
    float depth = texelFetch(depthBuffer, C, 0).r;
    vec3 currentWSPosition = reconstructWSPositionFromDepth(currentScreenCoord, depth, projInfo, clipInfo, cameraToWorld);
    return reverseReprojection(currentScreenCoord, currentWSPosition, ssV, previousBuffer, 
                         previousDepthBuffer, inverseBufferSize, clipInfo, projInfo, previousCameraToWorld, distance);
    
}






/** Requires all input buffers to be the same size as the output buffer

    Returns the reverse-reprojected value, and sets distance 
    to the WS distance from the expected WS reverse-reprojected position and the actual value.
*/
vec4 twoLayerReverseReprojection(vec2 currentScreenCoord,  sampler2D depthBuffer, 
                         sampler2D ssVelocityBuffer, vec2 ssVReadMultiplyFirst, vec2 ssVReadAddSecond, 
                         sampler2D previousBuffer, vec2 previousBufferInverseSize, sampler2D previousDepthBuffer, 
                         sampler2D peeledPreviousBuffer, sampler2D peeledPreviousDepthBuffer, vec2 inverseBufferSize,
                         vec3 clipInfo, vec4 projInfo, 
                         mat4x3 cameraToWorld, mat4x3 previousCameraToWorld, out float distance) {
    ivec2 C = ivec2(currentScreenCoord);
    vec2 ssV = texelFetch(ssVelocityBuffer, C, 0).rg * ssVReadMultiplyFirst + ssVReadAddSecond;
    float depth = texelFetch(depthBuffer, C, 0).r;
    vec3 currentWSPosition = reconstructWSPositionFromDepth(currentScreenCoord, depth, projInfo, clipInfo, cameraToWorld);
    return twoLayerReverseReprojection(currentScreenCoord, currentWSPosition, ssV, previousBuffer, 
                         previousDepthBuffer, peeledPreviousBuffer, peeledPreviousDepthBuffer, inverseBufferSize,
                         clipInfo, projInfo, previousCameraToWorld, distance);

}



#define viewMAD(m, v) (mat3(m) * (v) + (m)[3].xyz)

vec3 reproject(vec3 sceneSpace, bool hand) {
    vec3 prevScreenPos = hand ? vec3(0.0) : 
	cameraPosition - previousCameraPosition;
    prevScreenPos = sceneSpace + prevScreenPos;
    prevScreenPos = viewMAD(gbufferPreviousModelView, prevScreenPos);
    prevScreenPos = viewMAD(gbufferPreviousProjection, prevScreenPos) * (0.5 / -prevScreenPos.z) + 0.5;

    return prevScreenPos;
}

float checkerboard(in vec2 uv)
{
    vec2 pos = floor(uv);
  	return mod(pos.x + mod(pos.y, 2.0), 2.0);
}		




void temporal(inout vec3 indirectCurrent,inout vec4 historyGData,inout vec4 indirectHistory,vec3 fragpos,vec3 normal2, float z,vec2 texcoord , bool hand, vec3 inderectNoSSGI)
{


float checkerboard = checkerboard(gl_FragCoord.xy);


    vec3 scenePos   = viewMAD(gbufferModelViewInverse,fragpos);
    vec3 reprojection   = reproject(scenePos, hand);	

	
    bool offscreen      = clamp(reprojection,0,1) != reprojection;
    vec3 cameraMovement = mat3(gbufferModelView) * (cameraPosition - previousCameraPosition);
    vec4 currentGData   = vec4(normal2,0);	
         currentGData.rgb = currentGData.rgb * 2.0 - 1.0;
         currentGData.a  = length(fragpos);	
		
 	  vec4 velocity2  =	texture2D(colortex13,texcoord).rgba;	
	   	   velocity2.r = 	((velocity2.x/ld(z)*far)*0.000032);
	   	   velocity2.g = 	((velocity2.y/ld(z)*far)*0.00006);
//		   velocity2.xy = velocity2.xy*((velocity2.z/ld(z)*far)*0.1);	

	    reprojection.xy *= RENDER_SCALE;  
	if(velocity2.a >0.0)	reprojection.xy = texcoord - velocity2.rg;		
        historyGData        = texture2D(colortex9, reprojection.xy);
        historyGData.rgb    = historyGData.rgb * 2.0 - 1.0;


	//		  reprojection.xy +=   (velocity2)/RENDER_SCALE;
	//		  reprojection.xy =   (texcoord -velocity2)/RENDER_SCALE;
        float distanceDelta = distance(historyGData.a * far, currentGData.a) - abs(cameraMovement.z);
        float depthRejection = (offscreen) ? 0.0 : exp(-max(distanceDelta - 0.2, 0.0) * 3.0);
		
        indirectHistory     = texture2D(colortex12, reprojection.xy);
        indirectHistory.a   = clamp(indirectHistory.a,0,1);

        float normalWeight  = sqr(clamp(dot(historyGData.rgb, currentGData.rgb),0,1));


        float accumulationR0        = mix(0.1, 0.01, sqrt(indirectHistory.a));
        float accumulationWeight    = depthRejection;
			  accumulationWeight     *= normalWeight;
			  accumulationWeight     += clamp(((distance(indirectHistory.rgb,indirectCurrent.rgb)/luma(indirectHistory.rgb)) ) *(abs(cameraMovement.r)+abs(cameraMovement.g)+abs(cameraMovement.b)),0,1);	 			  
			  accumulationWeight      = clamp(1.0 - accumulationWeight,0,1);		
			  accumulationWeight      = max(accumulationWeight, accumulationR0);		

	vec3 closestToCamera = closestToCamera5taps(texcoord);
	vec3 fragposition = toScreenSpace(closestToCamera);			
			
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;
	//	 velocity = velocity2;	
	previousPosition.xy = texcoord + velocity;


		vec3 albedoCurrent0 = texture2D(colortex12, texcoord).rgb;
		vec3 albedoCurrent1 = texture2D(colortex12, texcoord + vec2(texelSize.x,texelSize.y)).rgb;
		vec3 albedoCurrent2 = texture2D(colortex12, texcoord + vec2(texelSize.x,-texelSize.y)).rgb;
		vec3 albedoCurrent3 = texture2D(colortex12, texcoord + vec2(-texelSize.x,-texelSize.y)).rgb;
		vec3 albedoCurrent4 = texture2D(colortex12, texcoord + vec2(-texelSize.x,texelSize.y)).rgb;
		vec3 albedoCurrent5 = texture2D(colortex12, texcoord + vec2(0.0,texelSize.y)).rgb;
		vec3 albedoCurrent6 = texture2D(colortex12, texcoord + vec2(0.0,-texelSize.y)).rgb;
		vec3 albedoCurrent7 = texture2D(colortex12, texcoord + vec2(-texelSize.x,0.0)).rgb;
		vec3 albedoCurrent8 = texture2D(colortex12, texcoord + vec2(texelSize.x,0.0)).rgb;
		
		//Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each channel in the 3x3 neighborhood
		vec3 cMax = max(max(max(albedoCurrent0,albedoCurrent1),albedoCurrent2),max(albedoCurrent3,max(albedoCurrent4,max(albedoCurrent5,max(albedoCurrent6,max(albedoCurrent7,albedoCurrent8))))));
		vec3 cMin = min(min(min(albedoCurrent0,albedoCurrent1),albedoCurrent2),min(albedoCurrent3,min(albedoCurrent4,min(albedoCurrent5,min(albedoCurrent6,min(albedoCurrent7,albedoCurrent8))))));



		vec3 albedoPrev = max(FastCatmulRom(colortex12, reprojection.xy,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
		vec3 albedoPrev2 = max(FastCatmulRom(colortex5, reprojection.xy/RENDER_SCALE,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
		vec3 finalcAcc = clamp(albedoPrev,cMin,cMax);		

		float isclamped = (clamp(clamp(((distance(albedoPrev,finalcAcc)/luma(albedoPrev))),0,1),0,1)*0.1);	 
		float isclamped2 = (((distance(albedoPrev2,finalcAcc)/luma(albedoPrev2)) *0.9) );	 
	//	float isclamped3 = (((distance(luma(albedoPrev2),amb)/luma(albedoPrev2)) *0.9) );	 
		float clamped = (dot(isclamped,isclamped2))*10 ;

		float acctest = distance(albedoPrev.rgb,finalcAcc.rgb)/luma(albedoPrev.rgb);

		      accumulationWeight = clamp(accumulationWeight + clamped ,0,1);	
		float accumulationWeight2 =  clamp(accumulationWeight+ ((abs(velocity2.r)+abs(velocity2.g)+abs(velocity2.z*0.5))*20),0.0,1);

        if (!offscreen) {
   		//	indirectCurrent.rgb     = mix(indirectCurrent.rgb, inderectNoSSGI.rgb, clamp( isclamped,0,1));
            indirectHistory.rgb     = mix(indirectHistory.rgb, indirectCurrent, accumulationWeight2);
            indirectHistory.a       = mix(0.0, clamp(indirectHistory.a + rcp(30),0,1), float(distanceDelta < 0.2));
        } else {
            indirectHistory         = vec4(indirectCurrent, 0.0);
        }	
	


        indirectCurrent     = indirectHistory.rgb;
        historyGData.rgb    = currentGData.rgb ;
        historyGData.a      = currentGData.a / far;

indirectCurrent = indirectCurrent;
historyGData = historyGData;
indirectHistory = indirectHistory;


}



//////////////////////////////SSGI//////////////////////////////
					

vec3 rtGI(vec3 normal,vec4 noise,vec3 fragpos, vec3 ambient, float translucent, vec3 torch, vec3 albedo, float amb,float z, vec4 dataUnpacked1, vec3 shadowCol,vec2 lightmap, bool emissive,bool hand, vec2 texcoord){

	vec4    historyGData    = vec4(1.0);
	vec4   indirectHistory = vec4(0.0);
	vec4    indirectCurrent = texture2D(colortex5,texcoord.xy/RENDER_SCALE).rgba;
	float    sceneDepth = texture2D(depthtex0,texcoord.xy).x;

 
    vec3 scenePos   = viewMAD(gbufferModelViewInverse,fragpos);

    vec3 reprojection   = reproject(scenePos, hand);	
	bool offscreen      = clamp(reprojection,0,1) != reprojection;

	int nrays = RAY_COUNT;
//	int nrays = clamp(int(clamp((lightmap.x),0,1)*6),2,6);
//	if (z > 0.50) nrays = 2;
//	if (z > 0.75) nrays = 1;
	float mixer = SSPTMIX1;
//	float mixer = SSPTMIX1*clamp(1-(z*10),0,1);
	float rej = 1;
	vec3 intRadiance = vec3(0.0);
	

	
	vec3 intRadiance2 = vec3(texture2D(colortex5,texcoord.xy/RENDER_SCALE).rgb);
	float occlusion = 0.0;
	float depthmask = ((z*z*z)*2);
	if (depthmask >1) nrays = 1;



	
		vec4 transparencies = texture2D(colortex2,texcoord);			
		bool istransparent = luma(transparencies.rgb) > 0.1;		

	for (int i = 0; i < nrays; i++){ 
	
	
		int seed = (frameCounter%40000)*nrays+i;
		vec2 ij = fract(R2_samples(seed) + noise.rg);
	//		 ij.xy *= clamp(1+lightmap.x,1.0,1);
		vec3 rayDir = normalize(cosineHemisphereSample(ij));
			
		rayDir = TangentToWorld(normal,rayDir);
		rayDir = mat3(gbufferModelView)*rayDir;

		
	
		vec3 rayHit = RT(rayDir, fragpos, fract(seed/1.6180339887 + noise.b), mat3(gbufferModelView)*normal,luma(transparencies.rgb),lightmap,emissive,hand,z);



		vec3 previousPosition   = reproject(mat3(gbufferModelViewInverse) * toScreenSpace(rayHit) + gbufferModelViewInverse[3].xyz, false);	
		
		if (rayHit.z < 1.0){
 
			if (previousPosition.x > 0.0 && previousPosition.y > 0.0 && previousPosition.x < 1.0 && previousPosition.x < 1.0){
				#ifdef NETHER
				intRadiance += ((texture2D(colortex5,previousPosition.xy).rgb*(1+(lightmap.x*4)))  + ambient*albedo*translucent) ;
				#else
				intRadiance += ((texture2D(colortex5,previousPosition.xy).rgb*(1+(lightmap.x*2)))  + ambient*albedo*translucent) ;
				#endif

		#ifdef ssgi_staturation
				float lum = luma(intRadiance);
				vec3 diff = intRadiance-lum;
				intRadiance = (intRadiance + diff*(0.1));
		#endif		
		torch = vec3(0);
				}
						
			else{

			
				intRadiance += ambient + ambient *translucent*albedo;
			
				}
					occlusion += 1;
				
		}		
		else {
						float bounceAmount = float(rayDir.y > 0.0);
			vec3 sky_c = skyCloudsFromTex(rayDir,colortex4).rgb*8/3./150. * bounceAmount;
				 sky_c *= clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0);		
				 sky_c *= lightmap.y;	

			intRadiance += sky_c;

		}
		

		
	}
	
	if (hand) occlusion =0.0;

//	return intRadiance/nrays + (1.0-occlusion/nrays)*torch*0;
	#ifdef NETHER
	intRadiance.rgb =  ((intRadiance  /nrays + (1.0-(occlusion*2)/nrays)* ( ( (torch*0.50) + (ambient*0.15) )) +shadowCol ));	
	#else
	intRadiance.rgb =  ((intRadiance  /nrays + (1.0-(occlusion*2)/nrays)* ( ( (torch*0.75) + (ambient*1.0) )) +shadowCol ));	
	#endif
	intRadiance2.rgb = (intRadiance2 );	
	


	vec3 closestToCamera = closestToCamera5taps(texcoord);
	vec3 fragposition = toScreenSpace(closestToCamera);			
			
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;

	previousPosition.xy = texcoord + velocity;
	   
	   
	   

	   


	vec3 albedoCurrent0 = texture2D(colortex12, reprojection.xy*RENDER_SCALE).rgb;
	vec3 albedoCurrent1 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent2 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent3 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(-texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent4 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(-texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent5 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(0.0,texelSize.y)).rgb;
	vec3 albedoCurrent6 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(0.0,-texelSize.y)).rgb;
	vec3 albedoCurrent7 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(-texelSize.x,0.0)).rgb;
	vec3 albedoCurrent8 = texture2D(colortex12, reprojection.xy*RENDER_SCALE + vec2(texelSize.x,0.0)).rgb;
	
	//Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each channel in the 3x3 neighborhood
	vec3 cMax = max(max(max(albedoCurrent0,albedoCurrent1),albedoCurrent2),max(albedoCurrent3,max(albedoCurrent4,max(albedoCurrent5,max(albedoCurrent6,max(albedoCurrent7,albedoCurrent8))))));
	vec3 cMin = min(min(min(albedoCurrent0,albedoCurrent1),albedoCurrent2),min(albedoCurrent3,min(albedoCurrent4,min(albedoCurrent5,min(albedoCurrent6,min(albedoCurrent7,albedoCurrent8))))));


	vec3 albedoPrev = max(FastCatmulRom(colortex12, reprojection.xy*RENDER_SCALE,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
	vec3 albedoPrev2 = max(FastCatmulRom(colortex5, reprojection.xy,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
	vec3 finalcAcc = clamp(albedoPrev,cMin,cMax);		



	float isclamped = (clamp(clamp(((distance(albedoPrev,finalcAcc)/luma(albedoPrev))),0,10),0,10));	 
	float isclamped2 = (((distance(albedoPrev2,finalcAcc)/luma(albedoPrev2)) *0.9) );	 
	float isclamped3 = (((distance(luma(albedoPrev2),amb)/luma(albedoPrev2)) *0.9) );	 
	float clamped = dot(isclamped,isclamped2);
	 
	 float weight = clamp(   (isclamped3)   ,0,1);
	 

	 

	if (hand) weight =1.0;
	if (hand) occlusion =0.0;
	if (emissive) weight =0.0;

//	gl_FragData[1].a = mix(texture2D(colortex12,reprojection.xy*RENDER_SCALE).a ,weight ,0.1);	
			
//	weight = clamp( (((texture2D(colortex12,reprojection.xy*RENDER_SCALE).a) +((edgemask*5-0.75)*0) ) +((isclamped*2)*clamp(length(velocity/texelSize),0.0,1.0))),0.5,1.0);
	weight = 0;
	float weight2 = clamp(weight-0.75,0.01,0.1);

  
	if (previousPosition.x < 0.0 || previousPosition.y < 0.0 || previousPosition.x > RENDER_SCALE.x || previousPosition.y > RENDER_SCALE.y) weight = 1;

//		intRadiance.rgb = invTonemap(mix( tonemap(intRadiance),tonemap(intRadiance2),clamp( ((weight2) +depthmask )  ,0.0,1.0)))*(1.0-(occlusion)/nrays);	
		
//		intRadiance.rgb = clamp(invTonemap(mix(tonemap(texture2D(colortex12,reprojection.xy*RENDER_SCALE).rgb), tonemap(intRadiance.rgb), weight  )),0.0,1000);


//	gl_FragData[1].rgb = (intRadiance);


	


		
	return vec3(intRadiance).rgb*(1.0-(occlusion)/nrays);




}



