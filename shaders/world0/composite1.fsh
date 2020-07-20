#version 120
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"


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
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex6;//Skybox
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth
uniform sampler2D texture;
uniform sampler2D normals;



uniform int heldBlockLightValue;
uniform int frameCounter;
uniform float frameTime;
uniform int isEyeInWater;
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




const vec2 poissonDisk[64] = vec2[64](
vec2(-0.613392, 0.617481),
vec2(0.170019, -0.040254),
vec2(-0.299417, 0.791925),
vec2(0.645680, 0.493210),
vec2(-0.651784, 0.717887),
vec2(0.421003, 0.027070),
vec2(-0.817194, -0.271096),
vec2(-0.705374, -0.668203),
vec2(0.977050, -0.108615),
vec2(0.063326, 0.142369),
 vec2(0.203528, 0.214331),
 vec2(-0.667531, 0.326090),
 vec2(-0.098422, -0.295755),
 vec2(-0.885922, 0.215369),
 vec2(0.566637, 0.605213),
 vec2(0.039766, -0.396100),
 vec2(0.751946, 0.453352),
 vec2(0.078707, -0.715323),
 vec2(-0.075838, -0.529344),
 vec2(0.724479, -0.580798),
 vec2(0.222999, -0.215125),
 vec2(-0.467574, -0.405438),
 vec2(-0.248268, -0.814753),
 vec2(0.354411, -0.887570),
 vec2(0.175817, 0.382366),
 vec2(0.487472, -0.063082),
 vec2(-0.084078, 0.898312),
 vec2(0.488876, -0.783441),
 vec2(0.470016, 0.217933),
 vec2(-0.696890, -0.549791),
 vec2(-0.149693, 0.605762),
 vec2(0.034211, 0.979980),
 vec2(0.503098, -0.308878),
 vec2(-0.016205, -0.872921),
 vec2(0.385784, -0.393902),
 vec2(-0.146886, -0.859249),
 vec2(0.643361, 0.164098),
 vec2(0.634388, -0.049471),
 vec2(-0.688894, 0.007843),
 vec2(0.464034, -0.188818),
 vec2(-0.440840, 0.137486),
 vec2(0.364483, 0.511704),
 vec2(0.034028, 0.325968),
 vec2(0.099094, -0.308023),
 vec2(0.693960, -0.366253),
 vec2(0.678884, -0.204688),
 vec2(0.001801, 0.780328),
 vec2(0.145177, -0.898984),
 vec2(0.062655, -0.611866),
 vec2(0.315226, -0.604297),
 vec2(-0.780145, 0.486251),
 vec2(-0.371868, 0.882138),
 vec2(0.200476, 0.494430),
 vec2(-0.494552, -0.711051),
 vec2(0.612476, 0.705252),
 vec2(-0.578845, -0.768792),
 vec2(-0.772454, -0.090976),
 vec2(0.504440, 0.372295),
 vec2(0.155736, 0.065157),
 vec2(0.391522, 0.849605),
 vec2(-0.620106, -0.328104),
 vec2(0.789239, -0.419965),
 vec2(-0.545396, 0.538133),
 vec2(-0.178564, -0.596057)
);

vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}


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







float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));

}


vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
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




float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}


vec4 blueNoise(vec2 coord){
  return texelFetch2D(colortex6, ivec2(coord)%512, 0);
}




void waterVolumetrics(inout vec3 inColor, vec3 rayStart, vec3 rayEnd, float estEndDepth, float estSunDepth, float rayLength, float dither, vec3 waterCoefs, vec3 scatterCoef, vec3 ambient, vec3 lightSource, float VdotL){
		inColor *= exp(-rayLength * waterCoefs);	//No need to take the integrated value
		int spCount = rayMarchSampleCount;
		vec3 start = vec3(0.0);
		vec3 end = vec3(0.0);
		vec3 dV = (end-start);
		//limit ray length at 32 blocks for performance and reducing integration error
		//you can't see above this anyway
		float maxZ = min(rayLength,32.0)/(1e-8+rayLength);
		dV *= maxZ;
		rayLength *= maxZ;
		estEndDepth *= maxZ;
		estSunDepth *= maxZ;
		vec3 absorbance = vec3(1.0);
		vec3 vL = vec3(0.0);
		float phase = phaseg(VdotL, Dirt_Mie_Phase);
		float expFactor = 11.0;
		for (int i=0;i<spCount;i++) {
			float d = (pow(expFactor, float(i+dither)/float(spCount))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
			float dd = pow(expFactor, float(i+dither)/float(spCount)) * log(expFactor) / float(spCount)/(expFactor-1.0);
			vec3 spPos = start.xyz + dV*d;
			//project into biased shadowmap space

			vec3 pos = vec3(spPos.xy, spPos.z);
			float sh = 1.0;
			if (abs(pos.x) < 1.0-0.5/2048. && abs(pos.y) < 1.0-0.5/2048){
				pos = pos*vec3(0.5,0.5,0.5/6.0)+0.5;
				sh =  0;
			}
			vec3 ambientMul = exp(-estEndDepth * d * waterCoefs * 1.1);
			vec3 sunMul = exp(-estSunDepth * d * waterCoefs);
			vec3 light = (sh * lightSource*8./150./3.0 * phase * sunMul + ambientMul * ambient)*scatterCoef;
			vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs *absorbance;
			absorbance *= exp(-dd * rayLength * waterCoefs);
		}
		inColor += vL;
}

float waterCaustics(vec3 wPos){
	vec2 pos = (wPos.xz + wPos.y)*4.0 ;
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
vec2 decodeVec1(float a){
    const vec2 constant1 = 65535. / vec2( 256., 65536.);
    const float constant2 = 256. / 255.;
    return fract( a * constant1 ) * constant2 ;
}
const vec2 shadowOffsets[6] = vec2[6](vec2(  0.5303,  0.5303 ),
vec2( -0.6250, -0.0000 ),
vec2(  0.3536, -0.3536 ),
vec2( -0.0000,  0.3750 ),
vec2( -0.1768, -0.1768 ),
vec2( 0.1250,  0.0000 ));

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

	if (z >=1.0) {

		vec3 color = vec3(0.0);
		vec4 cloud = texture2D_bicubic(colortex0,texcoord*CLOUDS_QUALITY);
		if (np3.y > 0.){
			color += stars(np3);
			color += drawSun(dot(lightCol.a*WsunVec,np3),0, lightCol.rgb/150.,vec3(0.0));
		}
		color += skyFromTex(np3,colortex4)/150. + toLinear(texture2D(colortex1,texcoord).rgb)/10.*4.0*ffstep(0.985,-dot(lightCol.a*WsunVec,np3));
		color = color*cloud.a+cloud.rgb;
		gl_FragData[0].rgb = clamp(fp10Dither(color*8./3.0,triangularize(noise)),0.0,65000.);
		//if (gl_FragData[0].r > 65000.) 	gl_FragData[0].rgb = vec3(0.0);
		vec4 trpData = texture2D(colortex7,texcoord);
		bool iswater = texture2D(colortex7,texcoord).a > 0.99;
		if (iswater){
			vec3 fragpos0 = toScreenSpace(vec3(texcoord-vec2(tempOffset)*texelSize*0.5,z0));


		}
	}
	//land
	else {
		p3 += gbufferModelViewInverse[3].xyz;

		vec4 trpData = texture2D(colortex7,texcoord);
		bool iswater = texture2D(colortex7,texcoord).a > 0.99;

		vec4 data = texture2D(colortex1,texcoord);
		vec4 entityg = texture2D(colortex7,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);
		vec2 lightmap = vec2(dataUnpacked1.yz);				
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool entity = abs(entityg.y) >0.9;
		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		float NdotL = dot(normal,WsunVec);


		float diffuseSun = clamp(NdotL,0.0,1);
		float shading = 1.0;




		vec3 filtered = vec3(1.412,1.0,0.0);
		if (!hand){
			filtered = texture2D(colortex3,texcoord).rgb;
		}

			if (translucent) {
			diffuseSun = mix(max(phaseg(dot(np3, WsunVec),0.1), 2.0*phaseg(dot(np3, WsunVec),0.1))*PI*PI*(1.0-filtered.y*0.5), diffuseSun, 0.3);
			filtered.y = filtered.y * 0.55+0.55;
		}

		vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.0));

		vec3 ambientLight = ambientUp*clamp(ambientCoefs.y,0.,1.);
		ambientLight += ambientDown*clamp(-ambientCoefs.y,0.,1.);
		ambientLight += ambientRight*clamp(ambientCoefs.x,0.,1.);
		ambientLight += ambientLeft*clamp(-ambientCoefs.x,0.,1.);
		ambientLight += ambientB*clamp(ambientCoefs.z,0.,1.);
		ambientLight += ambientF*clamp(-ambientCoefs.z,0.,1.);
		ambientLight *= (1.0+rainStrength*0.2);
		vec3 directLightCol = lightCol.rgb*((lightmap.y-0.9)*10);

		if (lightmap.y < 0.9) directLightCol = vec3(0.0);
		
		
		
		vec3 custom_lightmap = texture2D(colortex4,(lightmap*15.0+0.5+vec2(0.0,19.))*texelSize).rgb*4./150./3.;
		vec3 custom_lightmap2 = texture2D(colortex4,(lightmap*15.0+0.5+vec2(0.0,0.))*texelSize).rgb*4./150./3.;
		custom_lightmap.y *= filtered.y*0.9+0.1;

		if (emissive || (hand && heldBlockLightValue > 0.1))

			custom_lightmap.y = pow(clamp(albedo.r-0.35,0.0,1.0)/0.65*0.65+0.35,2.0)*2;	

		if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater ==1)){

			vec3 fragpos0 = toScreenSpace(vec3(texcoord-vec2(tempOffset)*texelSize*0.5,z0));
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
			float caustics = waterCaustics(mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition);
			directLightCol *= mix(caustics,1.0,exp(-0.1*estimatedSunDepth)*0.5+0.5);

			ambientLight *= exp(-totEpsilon*estimatedDepth*1.1)*0.8*8./150./3.;
			if (isEyeInWater == 0){
			
				ambientLight *= custom_lightmap.x/(10./150./3.);
				ambientLight += custom_lightmap.z;
			}
			else
				ambientLight += custom_lightmap.z * 70.0 * exp(-totEpsilon*16.0);

			ambientLight *= mix(caustics,1.0,0.9);
			if (emissive){ ambientLight += custom_lightmap.y;} 
						else{ ambientLight += custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B);}

			//combine all light sources
			
			
			gl_FragData[0].rgb = ((shading*diffuseSun)/pi*8./150./3.*(directLightCol.rgb*lightmap.yyy) + filtered.y*ambientLight)*albedo;
			
			
			//Bruteforce integration is probably overkill
			vec3 lightColVol = lightCol.rgb * (0.91-pow(1.0-WsunVec.y,5.0)*0.86);	//fresnel
			vec3 ambientColVol =  ambientUp*8./150./3.*0.84*2.0/pi / 240.0 * eyeBrightnessSmooth.y;
			if (isEyeInWater == 0)
				waterVolumetrics(gl_FragData[0].rgb, fragpos0, fragpos, estimatedDepth, estimatedSunDepth, Vdiff, noise, totEpsilon, scatterCoef, ambientColVol, lightColVol, dot(np3, WsunVec));

		}
		else {

			
				vec3 colorContrasted = (custom_lightmap2) *20;
				vec3 bright = clamp(colorContrasted - vec3(19),0,255);
			ambientLight = ambientLight * filtered.y* custom_lightmap.x + custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B) + custom_lightmap.z*vec3(0.9,1.0,1.5)*filtered.y;
			if (emissive) ambientLight = ((ambientLight *filtered.y* custom_lightmap.x + custom_lightmap.y + custom_lightmap.z*vec3(0.9,1.0,1.5))*filtered.y)*albedo.rgb+0.5;
			
			gl_FragData[0].rgb = ((bright.y*diffuseSun)/pi/150./3.0*(directLightCol.rgb*lightmap.yyy) + (ambientLight))*albedo;


		//	 gl_FragData[0].rgb = (lightmap.yyy-0.6)*4;
			
		}
	}

/* DRAWBUFFERS:36 */
}
