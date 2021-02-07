






vec3 RT(vec3 dir,vec3 position,float noise, vec3 N,float transparent){
	float stepSize = STEP_LENGTH;
	int maxSteps = STEPS;
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
		if(istranparent)  return vec3(spos.xy, invLinZ(sp))/vec3(RENDER_SCALE,1.0);					
		if( sp < currZ && abs(sp-ld(spos.z))/ld(spos.z) < 0.1) {
	
		if(istranparent)  return vec3(spos.xy, invLinZ(sp))/vec3(RENDER_SCALE,1.0);		
		
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
vec3 rtGI(vec3 normal,vec4 noise,vec3 fragpos, vec3 ambient, float translucent, vec3 torch, vec3 albedo, float amb,float z, vec4 dataUnpacked1, float edgemask){

	bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
	bool hand = abs(dataUnpacked1.w-0.75) <0.01;
	int nrays = RAY_COUNT;
//	if (z > 0.50) nrays = 2;
//	if (z > 0.75) nrays = 1;
	float mixer = SSPTMIX1;
	float rej = 1;
	vec3 intRadiance = vec3(0.0);
	float occlusion = 0.0;
	float depthmask = ((z*z*z)*2);
	if (depthmask >1) nrays = 1;

	vec2 texcoord = gl_FragCoord.xy*texelSize;	

	if (hand) edgemask = 1.0;
		vec4 normal2 = (texture2D(colortexA, texcoord));
		vec3 normal3 =  (texture2D(colortex8, texcoord)).rgb;
	//	normal = normal +normal3;
		vec4 transparencies = texture2D(colortex2,texcoord);			
		

	for (int i = 0; i < nrays; i++){ 
	
	
		int seed = (frameCounter%40000)*nrays+i;
		vec2 ij = fract(R2_samples(seed) + noise.rg);
		vec3 rayDir = normalize(cosineHemisphereSample(ij));
		rayDir = TangentToWorld(normal,rayDir);

		
		vec3 rayHit = RT(mat3(gbufferModelView)*rayDir, fragpos, fract(seed/1.6180339887 + noise.b), mat3(gbufferModelView)*normal,luma(transparencies.rgb));
		vec3 previousPosition = mat3(gbufferModelViewInverse) * toScreenSpace(rayHit) + gbufferModelViewInverse[3].xyz + cameraPosition-previousCameraPosition;
		previousPosition = mat3(gbufferPreviousModelView) * previousPosition + gbufferPreviousModelView[3].xyz;
		previousPosition.xy = projMAD(gbufferPreviousProjection, previousPosition).xy / -previousPosition.z * 0.5 + 0.5;	
		

		
		if (rayHit.z < 1.0){
 
			if (previousPosition.x > 0.0 && previousPosition.y > 0.0 && previousPosition.x < 1.0 && previousPosition.x < 1.0)
			
				intRadiance += (texture2D(colortex5,previousPosition.xy).rgb   + ambient*albedo*translucent) ;
			else
				intRadiance += ambient + ambient*translucent*albedo;
				occlusion += 1.5;
				
		}		
		else {

		
			intRadiance += ambient*SSPTambient;
		}
		

		
	}

		
			

	vec3 closestToCamera = closestToCamera5taps(texcoord);
	vec3 fragposition = toScreenSpace(closestToCamera);			
			
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;

	previousPosition.xy = texcoord + velocity;
	   
	   
	   

	   


	vec3 albedoCurrent0 = texture2D(colortexC, texcoord).rgb;
	vec3 albedoCurrent1 = texture2D(colortexC, texcoord + vec2(texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent2 = texture2D(colortexC, texcoord + vec2(texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent3 = texture2D(colortexC, texcoord + vec2(-texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent4 = texture2D(colortexC, texcoord + vec2(-texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent5 = texture2D(colortexC, texcoord + vec2(0.0,texelSize.y)).rgb;
	vec3 albedoCurrent6 = texture2D(colortexC, texcoord + vec2(0.0,-texelSize.y)).rgb;
	vec3 albedoCurrent7 = texture2D(colortexC, texcoord + vec2(-texelSize.x,0.0)).rgb;
	vec3 albedoCurrent8 = texture2D(colortexC, texcoord + vec2(texelSize.x,0.0)).rgb;
	
	//Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each channel in the 3x3 neighborhood
	vec3 cMax = max(max(max(albedoCurrent0,albedoCurrent1),albedoCurrent2),max(albedoCurrent3,max(albedoCurrent4,max(albedoCurrent5,max(albedoCurrent6,max(albedoCurrent7,albedoCurrent8))))));
	vec3 cMin = min(min(min(albedoCurrent0,albedoCurrent1),albedoCurrent2),min(albedoCurrent3,min(albedoCurrent4,min(albedoCurrent5,min(albedoCurrent6,min(albedoCurrent7,albedoCurrent8))))));
	if (hand) occlusion =0.0;

	intRadiance.rgb = intRadiance/nrays + (1.0-occlusion/nrays)*mix(vec3(0.0),torch+ambient,mixer);			
	vec3 albedoPrev = max(FastCatmulRom(colortexC, previousPosition.xy,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
	vec3 albedoPrev2 = max(FastCatmulRom(colortex5, previousPosition.xy/RENDER_SCALE,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
	vec3 finalcAcc = clamp(albedoPrev,cMin,cMax);		



	float isclamped = (clamp(clamp(((distance(albedoPrev,finalcAcc)/luma(albedoPrev))),0,10),0,10));	 
	float isclamped2 = (((distance(albedoPrev2,finalcAcc)/luma(albedoPrev2)) *0.9) );	 
	float isclamped3 = (((distance(luma(albedoPrev2),amb)/luma(albedoPrev2)) *0.9) );	 
	float clamped = dot(isclamped,isclamped2);
	 
	 float weight = clamp(   (isclamped3+edgemask)   ,0,1);
	 

	 

	if (hand) weight =10.0;
	if (hand) occlusion =0.0;
	if (emissive) weight =0.0;
	gl_FragData[1].a = mix(texture2D(colortexC,previousPosition.xy).a,weight,0.5);	
		
	  weight = clamp( ((texture2D(colortexC,previousPosition.xy).a) +(edgemask))+(isclamped*0.1)*clamp(length(velocity/texelSize),0.0,2.0)    ,0.0,1);	
	 gl_FragData[4].rgb = vec3(weight); 
  
	  if (previousPosition.x < 0.0 || previousPosition.y < 0.0 || previousPosition.x > RENDER_SCALE.x || previousPosition.y > RENDER_SCALE.y) weight = 1.0;
	  
		intRadiance.rgb = invTonemap(mix( tonemap(intRadiance),tonemap(mix(vec3(0.0),(torch+ambient)*SSPTMIX1,1)),clamp( ((weight*0.1) +depthmask )  ,0.0,1.0)));	 
		intRadiance.rgb = clamp(invTonemap(mix(tonemap(texture2D(colortexC,previousPosition.xy).rgb),tonemap(intRadiance.rgb), weight  )),0.0,100);
		


	gl_FragData[1].rgb = (intRadiance.rgb);	
				gl_FragData[6].rgb = vec3(intRadiance.rgb);	

		
	
		
	return vec3(intRadiance).rgb*(1.0-occlusion/(nrays*2));
//	return vec3(intRadiance).rgb;



}



