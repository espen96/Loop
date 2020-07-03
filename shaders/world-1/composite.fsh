#version 120
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
const bool shadowHardwareFiltering = true;

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
uniform vec3 fogColor; 
uniform sampler2D colortex0;//clouds
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex6;//Skybox
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth

uniform int heldBlockLightValue;
uniform int frameCounter;
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
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferPreviousModelView;



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

#include "/lib/color_transforms.glsl"
#include "/lib/util.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/encode.glsl"
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
float interleaved_gradientNoise(){
	vec2 coord = gl_FragCoord.xy;
	float noise = fract(52.9829189*fract(0.06711056*coord.x + 0.00583715*coord.y));
	return noise;
}
vec3 fp10Dither(vec3 color,float dither){
	const vec3 mantissaBits = vec3(6.,6.,5.);
	vec3 exponent = floor(log2(color));
	return color + dither*exp2(-mantissaBits)*exp2(exponent);
}





float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
	// l = (2*n)/(f+n-d(f-n))
	// f+n-d(f-n) = 2n/l
	// -d(f-n) = ((2n/l)-f-n)
	// d = -((2n/l)-f-n)/(f-n)

}
float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
}

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}


#define bayer4(a)   (bayer2( .5*(a))*.25+bayer2(a))
#define bayer8(a)   (bayer4( .5*(a))*.25+bayer2(a))
#define bayer16(a)  (bayer8( .5*(a))*.25+bayer2(a))
#define bayer32(a)  (bayer16(.5*(a))*.25+bayer2(a))
#define bayer64(a)  (bayer32(.5*(a))*.25+bayer2(a))
#define bayer128(a) fract(bayer64(.5*(a))*.25+bayer2(a)+tempOffsets)
float rayTraceShadow(vec3 dir,vec3 position,float dither,float translucent){

    const float quality = 16.;
    vec3 clipPosition = toClipSpace3(position);
	//prevents the ray from going behind the camera
	float rayLength = ((position.z + dir.z * far*sqrt(3.)) > -near) ?
       (-near -position.z) / dir.z : far*sqrt(3.);
    vec3 direction = toClipSpace3(position+dir*rayLength)-clipPosition;  //convert to clip space
    direction.xyz = direction.xyz/max(abs(direction.x)/texelSize.x,abs(direction.y)/texelSize.y);	//fixed step size




    vec3 stepv = direction *3. * clamp(MC_RENDER_QUALITY,1.,2.0);

	vec3 spos = clipPosition+vec3(TAA_Offset*vec2(texelSize.x,texelSize.y)*0.5,0.0)+stepv*dither;





	for (int i = 0; i < int(quality); i++) {
		spos += stepv;

		float sp = texture2D(depthtex1,spos.xy).x;
        if( sp < spos.z) {

			float dist = abs(linZ(sp)-linZ(spos.z))/linZ(spos.z);

			if (dist < 0.01 ) return translucent*exp2(position.z/8.);



	}

	}
    return 1.0;
}



float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}


vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter,float distort)
{
		float alpha0 = sampleNumber/nb;
    float alpha = (sampleNumber+jitter)/nb;
    float angle = jitter*6.28 + alpha * 4.0 * 6.28;

    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*sqrt(alpha);
}



vec3 BilateralFiltering(sampler2D tex, sampler2D depth,vec2 coord,float frDepth,float maxZ){
  vec4 sampled = vec4(texelFetch2D(tex,ivec2(coord),0).rgb,1.0);

  return vec3(sampled.x,sampled.yz/sampled.w);
}
float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}

vec3 toShadowSpaceProjected(vec3 p3){
    p3 = mat3(gbufferModelViewInverse) * p3 + gbufferModelViewInverse[3].xyz;
    p3 = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
    p3 = diagonal3(shadowProjection) * p3 + shadowProjection[3].xyz;

    return p3;
}

#define nether
#include "/lib/sspt.glsl"


vec2 tapLocation(int sampleNumber, float spinAngle,int nb, float nbRot,float r0)
{
    float alpha = (float(sampleNumber*1.0f + r0) * (1.0 / (nb)));
    float angle = alpha * (nbRot * 6.28) + spinAngle*6.28;

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}
void ssao(inout float occlusion,vec3 fragpos,float mulfov,float dither,vec3 normal)
{
	ivec2 pos = ivec2(gl_FragCoord.xy);
	const float tan70 = tan(70.*3.14/180.);
	float mulfov2 = gbufferProjection[1][1]/tan70;

	const float PI = 3.14159265;
	const float samplingRadius = 0.712;
	float angle_thresh = 0.05;




	float rd = mulfov2*0.05;
	//pre-rotate direction
	float n = 0.;

	occlusion = 0.0;

	vec2 acc = -vec2(TAA_Offset)*texelSize*0.5;
	float mult = (dot(normal,normalize(fragpos))+1.0)*0.5+0.5;

	vec2 v = fract(vec2(dither,interleaved_gradientNoise()) + (frameCounter%10000) * vec2(0.75487765, 0.56984026));
	for (int j = 0; j < SSAO_SAMPLES+2 ;j++) {
			vec2 sp = tapLocation(j,v.x,SSAO_SAMPLES+2,2.,v.y);
			vec2 sampleOffset = sp*rd;
			ivec2 offset = ivec2(gl_FragCoord.xy + sampleOffset*vec2(viewWidth,viewHeight));
			if (offset.x >= 0 && offset.y >= 0 && offset.x < viewWidth && offset.y < viewHeight ) {
				vec3 t0 = toScreenSpace(vec3(offset*texelSize+acc+0.5*texelSize,texelFetch2D(depthtex1,offset,0).x));

				vec3 vec = t0.xyz - fragpos;
				float dsquared = dot(vec,vec);
				if (dsquared > 1e-5){
					if (dsquared < fragpos.z*fragpos.z*0.05*0.05*mulfov2*2.*1.412){
						float NdotV = clamp(dot(vec*inversesqrt(dsquared), normalize(normal)),0.,1.);
						occlusion += NdotV;
					}
					n += 1.0;
				}
			}
		}



		occlusion = clamp(1.0-occlusion/n*2.0,0.,1.0);
		//occlusion = mult;

}
void main() {
	float dirtAmount = Dirt_Amount;
	vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
	vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
	vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
	vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B) / pi;
	float z0 = texture2D(depthtex0,texcoord).x;
	float z = texture2D(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();

	vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(tempOffset)*texelSize*0.5,z));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normVec(p3);

	//sky
	if (z >=1.0) {
		vec3 color = vec3(1.0,0.4,0.12)* vec3(1.5,1.2,1.05)/4000.0*150.0*0.25;
		gl_FragData[0].rgb = clamp(fp10Dither(color*8./3. * (1.0-rainStrength*0.4),triangularize(noise)),0.0,65000.);
		//if (gl_FragData[0].r > 65000.) 	gl_FragData[0].rgb = vec3(0.0);
		vec4 trpData = texture2D(colortex7,texcoord);
		bool iswater = texture2D(colortex7,texcoord).a > 0.99;
		if (iswater){
			vec3 fragpos0 = toScreenSpace(vec3(texcoord-vec2(tempOffset)*texelSize*0.5,z0));
			float Vdiff = distance(fragpos,fragpos0);
			float VdotU = np3.y;
			float estimatedDepth = Vdiff * abs(VdotU);	//assuming water plane
			float estimatedSunDepth = estimatedDepth/abs(WsunVec.y); //assuming water plane

			vec3 lightColVol = lightCol.rgb * (0.91-pow(1.0-WsunVec.y,5.0)*0.86);	//fresnel
			vec3 ambientColVol = ambientUp*8./150./3.*0.84*2.0/pi * eyeBrightnessSmooth.y / 240.0;
		}
	}
	//land
	else {
		p3 += gbufferModelViewInverse[3].xyz;

		vec4 trpData = texture2D(colortex7,texcoord);
		bool iswater = texture2D(colortex7,texcoord).a > 0.99;

		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		vec4 entityg = texture2D(colortex7,texcoord);
		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);

		vec2 lightmap = dataUnpacked1.yz;
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool emissive = abs(dataUnpacked1.w-0.9) >0.4;
		bool entity = abs(entityg.y) >0.9;
		vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));

		vec3 ambientLight = ambientUp*clamp(ambientCoefs.y,0.,1.);
		ambientLight += ambientDown*clamp(-ambientCoefs.y,0.,1.);
		ambientLight += ambientRight*clamp(ambientCoefs.x,0.,1.);
		ambientLight += ambientLeft*clamp(-ambientCoefs.x,0.,1.);
		ambientLight += ambientB*clamp(ambientCoefs.z,0.,1.);
		ambientLight += ambientF*clamp(-ambientCoefs.z,0.,1.);
		vec3 directLightCol = lightCol.rgb;
		
		vec3 custom_lightmap = texture2D(colortex4,(lightmap*15.0+0.5+vec2(0.0,19.))*texelSize).rgb*8./150./3.;
		if (emissive || (hand && heldBlockLightValue > 0.1)) custom_lightmap.y = pow(clamp(albedo.r-0.35,0.0,1.0)/0.65*0.65+0.35,2.0)*10;
		
		
#ifndef SSPT		
				
				if (emissive)  {ambientLight = (ambientLight * custom_lightmap.x + custom_lightmap.y + custom_lightmap.z);
				}
						else{
		ambientLight = ambientLight * custom_lightmap.x + custom_lightmap.y*vec3(N_TORCH_R,N_TORCH_G,N_TORCH_B) + custom_lightmap.z;}

		
		
		//combine all light sources
		float ao = 1.0;
		if (!hand)
		{
		#ifdef SSAO
			ssao(ao,fragpos,1.0,noise,decode(dataUnpacked0.yw));
		#endif
		}
		gl_FragData[0].rgb = ambientLight*albedo*ao;
		
		
#else	



				if (entity || emissive) { ambientLight = ambientLight * custom_lightmap.x + custom_lightmap.y*vec3(N_TORCH_R,N_TORCH_G,N_TORCH_B) + custom_lightmap.z;
				if (emissive)  ambientLight = (ambientLight * custom_lightmap.x + custom_lightmap.y + custom_lightmap.z)*albedo;
				}
						else{
		
			ambientLight = (ambientLight * custom_lightmap.x + custom_lightmap.y*vec3(N_TORCH_R,N_TORCH_G,N_TORCH_B) + custom_lightmap.z)/2.5;
			

			
			ambientLight += (rtGI(normal, noise, fragpos)*20.0/150./3.0 ) *((ambientLight.y*2));  }
			
		//combine all light sources
		float ao = 1.0;
		if (!hand)
		{
		#ifdef SSAO
			ssao(ao,fragpos,1.0,noise,decode(dataUnpacked0.yw));
		#endif
		}
		gl_FragData[0].rgb = ambientLight*ao;
		

#endif   

		#ifdef SSAO
			ssao(ao,fragpos,1.0,noise,decode(dataUnpacked0.yw));
		#endif
		

		vec3 ambientLight3 = ambientUp*clamp(ambientCoefs.y,0.,1.);
		ambientLight3 += ambientDown*clamp(-ambientCoefs.y,0.,1.);
		ambientLight3 += ambientRight*clamp(ambientCoefs.x,0.,1.);
		ambientLight3 += ambientLeft*clamp(-ambientCoefs.x,0.,1.);
		ambientLight3 += ambientB*clamp(ambientCoefs.z,0.,1.);
		ambientLight3 += ambientF*clamp(-ambientCoefs.z,0.,1.);

			vec3 ambientLight2 = ambientLight3 * custom_lightmap.x + custom_lightmap.y*vec3(N_TORCH_R,N_TORCH_G,N_TORCH_B) + custom_lightmap.z;
			if (emissive) ambientLight2 = (ambientLight3 * custom_lightmap.x + custom_lightmap.y + custom_lightmap.z)*albedo;	  
			gl_FragData[1].rgb = ambientLight2.rgb*ao;}

/* DRAWBUFFERS:36 */
}
