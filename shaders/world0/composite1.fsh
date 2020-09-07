#version 120
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
const bool shadowHardwareFiltering = true;

const float eyeBrightnessHalflife = 5.0f;

varying vec2 texcoord;

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

uniform sampler2D colortex0;//clouds
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex2;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortex3;
uniform sampler2D colortex7;
uniform sampler2D colortex5;
uniform sampler2D colortex6;//Skybox
uniform sampler2D depthtex2;//depth
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth
uniform sampler2D texture;
uniform sampler2D normals;
uniform sampler2DShadow shadow;
uniform vec3 fogColor;


uniform sampler2DShadow shadowtex1;
uniform sampler2DShadow shadowtex0;

						   

uniform sampler2D shadowcolor1;
uniform sampler2DShadow shadowcolor0;

uniform int heldBlockLightValue;
uniform int frameCounter;
uniform float frameTime;
uniform int isEyeInWater;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjectionInverse;
uniform float far;
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
uniform int entityId;
uniform int blockEntityId;
uniform int worldTime;

uniform vec2 texelSize;
uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;
uniform vec3 cameraPosition;
uniform int framemod8;
uniform vec3 sunVec;
uniform ivec2 eyeBrightnessSmooth;



vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}

#include "/lib/res_params.glsl"
#include "/lib/waterOptions.glsl"
#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/util.glsl"
#include "/lib/encode.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/stars.glsl"
#include "/lib/volumetricClouds.glsl"
#include "/lib/waterBump.glsl"

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


float interleaved_gradientNoise2(){
	vec2 coord = gl_FragCoord.xy;
	float noise = fract(52.9829189*fract(0.06711056*coord.x + 0.00583715*coord.y));
	return noise;
}




float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}


vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

float rayTraceShadow(vec3 dir,vec3 position,float dither){

    const float quality = 16.;
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

vec2 tapLocation4(float sampleNumber, float spinAngle,float nb,float nbRot)
{
    float alpha = float(sampleNumber + spinAngle/6.28) /nb;
    float angle = alpha * (nbRot * 6.28) + spinAngle;

    float ssR = alpha*alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}

float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
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
vec2 tapLocation2(int sampleNumber, float spinAngle,int nb, float nbRot,float r0)
{
    float alpha = (float(sampleNumber*1.0f + r0) * (1.0 / (nb)));
    float angle = alpha * (nbRot * 6.28) + spinAngle*6.28;

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}


float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}


vec4 blueNoise(vec2 coord){
  return texelFetch2D(colortex6, ivec2(coord*blueNoise())%512, 0);
}


vec3 toShadowSpaceProjected(vec3 p3){
    p3 = mat3(gbufferModelViewInverse) * p3 + gbufferModelViewInverse[3].xyz;
    p3 = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
    p3 = diagonal3(shadowProjection) * p3 + shadowProjection[3].xyz;

    return p3;
}



#include "/lib/sspt.glsl"

#ifdef PBR
#include "/lib/pbr.glsl"
#endif

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
				pos = pos*vec3(0.5,0.5,0.5/6.0)+0.5;
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

#ifdef SSAO
void ssao(inout float occlusion,vec3 fragpos,float mulfov,float dither,vec3 normal)
{
	ivec2 pos = ivec2(gl_FragCoord.xy);
	const float tan70 = tan(70.*3.14/180.);
	float mulfov2 = gbufferProjection[1][1]/tan70;

	const float PI = 3.14159265;
	const float samplingRadius = 0.712;
	float angle_thresh = 0.05;
	float maxR2 = fragpos.z*fragpos.z*mulfov2*2.*1.412/50.0;



	float rd = mulfov2*0.04;
	//pre-rotate direction
	float n = 0.;

	occlusion = 0.0;

	vec2 acc = -vec2(TAA_Offset)*texelSize*0.5;
	float mult = (dot(normal,normalize(fragpos))+1.0)*0.5+0.5;

	vec2 v = fract(vec2(dither,R2_dither()) + (frameCounter%10000) * vec2(0.75487765, 0.56984026));
	for (int j = 0; j < 7 ;j++) {

			vec2 sp = tapLocation2(j,v.x,7,88.,v.y);
			vec2 sampleOffset = sp*rd;
			ivec2 offset = ivec2(gl_FragCoord.xy + sampleOffset*vec2(viewWidth,viewHeight*aspectRatio)*RENDER_SCALE);
			if (offset.x >= 0 && offset.y >= 0 && offset.x < viewWidth*RENDER_SCALE.x && offset.y < viewHeight*RENDER_SCALE.y ) {
				vec3 t0 = toScreenSpace(vec3(offset*texelSize+acc+0.5*texelSize,texelFetch2D(depthtex1,offset,0).x) * vec3(1.0/RENDER_SCALE, 1.0));

				vec3 vec = t0.xyz - fragpos;
				float dsquared = dot(vec,vec);
				if (dsquared > 1e-5){
					if (dsquared < maxR2){
						float NdotV = clamp(dot(vec*inversesqrt(dsquared), normalize(normal)),0.,1.);
						occlusion += NdotV * clamp(1.0-dsquared/maxR2,0.0,1.0);
					}
					n += 1.0;
				}
			}
		}



		occlusion = clamp(1.0-occlusion/n*1.6,0.,1.0);
		//occlusion = mult;

}
#endif






void main() {




	vec2 texcoord = gl_FragCoord.xy*texelSize;							 
	float masks = texture2D(colortex3,texcoord).a;

	float dirtAmount = Dirt_Amount;
	vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
	vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
	vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
	vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B) / pi;
	float z0 = texture2D(depthtex0,texcoord).x;
	float z = texture2D(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();

	vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normVec(p3);

	
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

		bool iswater = texture2D(colortex3,texcoord).a > 0.9;
		if (iswater){
		gl_FragData[0].a = masks;
			vec3 fragpos0 = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z0));
			float Vdiff = distance(fragpos,fragpos0);
			float VdotU = np3.y;
			float estimatedDepth = Vdiff * abs(VdotU);	//assuming water plane
			float estimatedSunDepth = estimatedDepth/abs(WsunVec.y); //assuming water plane

			vec3 lightColVol = lightCol.rgb * (0.91-pow(1.0-WsunVec.y,5.0)*0.86);	//fresnel
			vec3 ambientColVol = ambientUp*8./150./3.*0.84*2.0/pi * eyeBrightnessSmooth.y / 240.0;
			if (isEyeInWater == 0)
				waterVolumetrics(gl_FragData[0].rgb, fragpos0, fragpos, estimatedDepth, estimatedSunDepth, Vdiff, noise, totEpsilon, scatterCoef, ambientColVol, lightColVol, dot(np3, WsunVec));
		}
	}
	//land
	else {
		p3 += gbufferModelViewInverse[3].xyz;


		bool iswater = texture2D(colortex3,texcoord).a > 0.9;

		vec4 data = texture2D(colortex1,texcoord);

		vec2 sp1 = decodeVec2(texture2D(colortex3,texcoord).g);
		vec2 sp2 = decodeVec2(texture2D(colortex3,texcoord).b);
		vec3 tester = texture2D(colortex7,texcoord).rgb;
		

		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);
		vec3 normal2 = decode(dataUnpacked0.yw);
		vec3 reflected = vec3(0.0);
		vec2 lightmap = vec2(dataUnpacked1.yz);			

		
		#ifdef GI
		vec3 rsm = texture2D(colortex7,texcoord).rgb;
		#endif
		#ifdef PBR
		vec4 specular = vec4(sp1,sp2);
		#endif
		
		
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;	// Strong translucency
		bool translucent2 = abs(dataUnpacked1.w-0.6) <0.01;	// Weak translucency
		bool glass = texture2D(colortex2,texcoord).a >=0.01;													  
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool entity = (masks) <=0.10 && (masks) >=0.09;
		bool lightning = (masks) <=0.20 && (masks) >=0.19;
		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		float NdotL = dot(normal,WsunVec);
		float diffuseSun = clamp(NdotL,0.,1.0);
		
		float ao = 1.0;
		
		#ifndef SSPT
		if (!hand)
		{
		#ifdef SSAO
			ssao(ao,fragpos,1.0,noise,decode(dataUnpacked0.yw));
		#endif
		}
		#endif
		
		
		
#ifdef PBR		
//////////////////////////////PBR//////////////////////////////		

		vec3 mat_data = vec3(0.0);		

		bool is_metal = false;
		
				mat_data = labpbr(specular, is_metal);
		float roughness = mat_data.x;
		float F0 = mat_data.y;
		float f0 = (F0*(1.0-gl_FragData[0].a));
		vec3 sky = skyFromTex(np3,colortex4)/150. + toLinear(texture2D(colortex1,texcoord).rgb)/10.*4.0*ffstep(0.985,-dot(lightCol.a*WsunVec,np3));


		float fresnel = pow(clamp(1.0 + dot(normal, normalize(fragpos.xyz)), 0.0, 1.0), 5.0);
		fresnel = mix(f0,1.0,fresnel);

	
		vec3 ssptr = SSPTR(normal2, blueNoise(gl_FragCoord.xy), fragpos, roughness, f0, fresnel, sky);
		     reflected = ssptr.rgb*fresnel;

//////////////////////////////PBR//////////////////////////////		
#endif


float weight = 0.0;
vec3 indirectLight = vec3(0.0);
vec3 shadowColBase = vec3(0.0);
vec3 shadowCol = vec3(0.0);
vec3 rsmfinal = vec3(0.0);
vec3 caustic = vec3(0.0);
float shadow1 = 0.0;
float shadow0 = 0.0;

		
		vec3 filtered = vec3(1.412,1.0,0.0);
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
		if (diffuseSun > 0.001 && (abs(pShadow) < 0.99 || length(fragpos) < 3.0 )) {
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
				const float threshMul = max(2048.0/shadowMapResolution*shadowDistance/128.0,0.95);
				float distortThresh = (sqrt(1.0-diffuseSun*diffuseSun)/diffuseSun+0.7)/distortFactor;
				#ifdef Variable_Penumbra_Shadows
				float diffthresh = distortThresh/6000.0*threshMul;
				#else
				float diffthresh = translucent? 0.0001 : distortThresh/6000.0*threshMul;
				#endif
				projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);
				shading = 0.0;
				for(int i = 0; i < SHADOW_FILTER_SAMPLE_COUNT; i++){
					vec2 offsetS = tapLocation(i,SHADOW_FILTER_SAMPLE_COUNT, 0.0,noise,0.0);

					float weight = 1.0+(i+noise)*rdMul/SHADOW_FILTER_SAMPLE_COUNT*shadowMapResolution;
					float isShadow = shadow2D(shadow,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x;
					shading += isShadow/SHADOW_FILTER_SAMPLE_COUNT;
				}
			}
		}
		
		
			vec3 closestToCamera = vec3(texcoord,texture2D(depthtex2,texcoord).x);

			
			
#ifdef GI			
	//reproject previous frame
	vec3 fragposition =	toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;
	previousPosition.xy = texcoord + velocity;
			
		
	//	rsm = mix(texture2D(colortex5,previousPosition.xy).rgb,rsm,0.99);
		

		if(!hand ||!translucent || !iswater ||!translucent2 || lightmap.y>0) rsmfinal += rsm;

#endif		
		
		

		//custom shading model for translucent objects
		#ifdef Variable_Penumbra_Shadows
		if (translucent) {
			sssAmount = 0.5;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26/4.0*sssAmount,0.0,1.0);
			SSS *= scattering;
			shading *= 1.0 - sssAmount;
		}

		if (translucent2) {
			sssAmount = 0.2;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26/4.0*sssAmount,0.0,1.0);
			SSS *= scattering;
			shading *= 1.0 - sssAmount;
		}

		#endif

		if (diffuseSun*shading > 0.001 ){
			#ifdef SCREENSPACE_CONTACT_SHADOWS
				vec3 vec = lightCol.a*sunVec;
				float screenShadow = rayTraceShadow(vec,fragpos,noise);
				shading = min(screenShadow, shading);
			#endif

		#ifdef CAVE_LIGHT_LEAK_FIX
			shading = mix(0.0, shading, clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0));
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

		vec3 ambientLight = ambientUp*mix(clamp(ambientCoefs.y,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientDown*mix(clamp(-ambientCoefs.y,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientRight*mix(clamp(ambientCoefs.x,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientLeft*mix(clamp(-ambientCoefs.x,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientB*mix(clamp(ambientCoefs.z,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientF*mix(clamp(-ambientCoefs.z,0.,1.), 1.0/6.0, sssAmount);

		vec3 directLightCol = lightCol.rgb;
		vec3 custom_lightmap = texture2D(colortex4,(lightmap*15.0+0.5+vec2(0.0,19.))*texelSize).rgb*8./150./3.;

		float alblum = clamp(luma(albedo),0.30,0.35)+( ambientLight.y*(0.0005 + (rainStrength*0.001) ) ); 	


		
    #ifdef PBR
	#ifdef EMISSIVES
		if (!iswater){		
		float emissive3 = float(specular.a > 1.98 && specular.a < 2.02) * 0.25;
		float emissive2 = mix(specular.a < 1.0 ? specular.a : 0.0, 1.0, emissive3);
		
		custom_lightmap.y += clamp(emissive2,0.0,1.0)*2;	}
		emissive = abs(specular.a) >0.01;
		
	#endif
	#else

		if (emissive || (hand && heldBlockLightValue > 0.1)) custom_lightmap.y =  float (pow(clamp(alblum-0.35,0.0,1.0)/0.1*0.65+0.35,2.0))*10;

	#endif
	
	

		if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater ==1)){

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

			float estimatedSunDepth = estimatedDepth/abs(WsunVec.y); //assuming water plane
			directLightCol *= exp(-totEpsilon*estimatedSunDepth)*(0.91-pow(1.0-WsunVec.y,5.0)*0.86);
			float caustics = waterCaustics(mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition, WsunVec);
			directLightCol *= mix(caustics*0.5+0.5,1.0,exp(-estimatedSunDepth/3.0));

			ambientLight *= exp(-totEpsilon*estimatedDepth*1.1)*0.8*8./150./3.;
			if (isEyeInWater == 0){
			
				ambientLight *= custom_lightmap.x/(8./150./3.);
				ambientLight += custom_lightmap.z;
			}
			else
				ambientLight += custom_lightmap.z * 70.0 * exp(-totEpsilon*16.0);

			float causticsAmbient = waterCaustics(mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition, vec3(1.0, 1.0, 1.0));
			ambientLight *= mix(causticsAmbient,1.0,0.85);
			if (emissive){ ambientLight += custom_lightmap.y;} 
						else{ ambientLight += custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B);}

			//combine all light sources
			gl_FragData[0].rgb = ((shading*diffuseSun + SSS)/pi*8./150./3.*directLightCol.rgb + ambientLight)*albedo;

			//Bruteforce integration is probably overkill
			vec3 lightColVol = lightCol.rgb * (0.91-pow(1.0-WsunVec.y,5.0)*0.86);	//fresnel
			vec3 ambientColVol =  ambientUp*8./150./3.*0.84*2.0/pi / 240.0 * eyeBrightnessSmooth.y;
			if (isEyeInWater == 0)
				waterVolumetrics(gl_FragData[0].rgb, fragpos0, fragpos, estimatedDepth, estimatedSunDepth, Vdiff, noise, totEpsilon, scatterCoef, ambientColVol, lightColVol, dot(np3, WsunVec));

		}
		else {				
		
		shadowCol *= (1-shading);		
		if (lightning) ambientLight *= vec3(2.0);	

		
			#ifdef SPEC_SCREENSPACE_REFLECTIONS
				if (!entity) albedo.rgb += reflected.rgb*(shading*diffuseSun)/pi;
			#endif
			
			
			#ifndef SSPT


				 ambientLight = ambientLight* custom_lightmap.x + custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B) +((rsmfinal*directLightCol.rgb/pi*8./150./3.)*lightmap.y);
			if (emissive)  ambientLight = ambientLight* custom_lightmap.x + custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*albedo.rgb+0.3;

			#else
			
		  	if(!hand && !emissive){ambientLight = rtGI(normal, blueNoise(gl_FragCoord.xy), fragpos, ambientLight* custom_lightmap.x, sssAmount, custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B)+((rsmfinal*directLightCol.rgb/pi*8./150./3.)*lightmap.y), normalize(albedo+1e-5)*0.7);
		}
			else{	
				
				ambientLight = ambientLight * filtered.y* custom_lightmap.x + custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B) + custom_lightmap.z*vec3(0.9,1.0,1.5)*filtered.y;
				if (emissive)  ambientLight = ambientLight* custom_lightmap.x + custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*albedo.rgb+0.3;


		
		 
}
			#endif			
			//combine all light sources

		    
			gl_FragData[0].rgb = ((shading * diffuseSun + SSS)/pi*8./150./3.*(directLightCol.rgb*lightmap.yyy) + ambientLight)*albedo*ao;
		//	gl_FragData[0].rgb = vec3(cloudSpeed);		


		
		//	gl_FragData[0].rgb = cloudSpeed2;				

	
			
		}
		
}
	

gl_FragData[0].a = masks;	

	
	
/* DRAWBUFFERS:3 */
}
