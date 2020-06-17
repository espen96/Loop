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

uniform sampler2DShadow shadow;

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
uniform int worldTime;

uniform vec2 texelSize;
uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;
uniform vec3 cameraPosition;
uniform int framemod8;
uniform vec3 sunVec;
uniform ivec2 eyeBrightnessSmooth;
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)



vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}




#include "/lib/color_transforms.glsl"
#include "/lib/encode.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/stars.glsl"



vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}

#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
float triangularize(float dither)
{
    float center = dither*2.0-1.0;
    dither = center*inversesqrt(abs(center));
    return clamp(dither-fsign(center),0.0,1.0);
}

vec3 fp10Dither(vec3 color,float dither){
	const vec3 mantissaBits = vec3(6.,6.,5.);
	vec3 exponent = floor(log2(color));
	return color + dither*exp2(-mantissaBits)*exp2(exponent);
}




float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}


#include "/lib/blur.glsl"

void main() {


	float z0 = texture2D(depthtex0,texcoord).x;
	float z = texture2D(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();

	vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(tempOffset)*texelSize*0.5,z));
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
		
		vec4 entityg = texture2D(colortex7,texcoord);
		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool entity = abs(entityg.y) >0.999;
		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		vec3 filtered = texture2D(colortex3,texcoord).rgb;



		 {
float Depth = texture2D(depthtex0, texcoord).x;
vec2 offset2 = vec2(1,0.99999);
vec3 blur = texture2D(colortex3, texcoord).xyz;
vec3 blur2 = texture2D(colortex3, texcoord).xyz;
vec3 blur3 = texture2D(colortex3, texcoord).xyz;
#ifndef RT_FILTER
blur = filtered.rgb;
blur2 = filtered.rgb;
blur3 = filtered.rgb;
#else
if (Depth < 1.0){
	Depth = ld(Depth);
    
	


	blur2 = mix((ssaoVL_blur(texcoord,vec2(0.5,0.1),Depth*far)),(ssaoVL_blur(texcoord,vec2(0.1,0.5),Depth*far)),0.5);
	blur3 = (blur2/filtered)/4;
	blur = mix(blur2,filtered,blur3*0.5);
	//blur = ssaoVL_blur(texcoord,vec2(0.0,0.5),Depth*far);


}
  float lum = luma(filtered);
  vec3 diff = filtered-lum;
  vec3 filtered2 = blur + diff*(-lum*(-0.0) + 0.0);
#endif		    
		    gl_FragData[0].rgb = filtered.rgb;	
		
		   #ifdef SSPT

		    gl_FragData[0] = vec4((blur.xyz)*albedo,1.0);
	
			if (iswater){ 
			gl_FragData[0].rgb = filtered.rgb;}
			if (isEyeInWater == 1 || entity) { gl_FragData[0].rgb = filtered.rgb * albedo;}



			

			#endif
			
		    //gl_FragData[0] = vec4((blur.rgb),1.0);
			//gl_FragData[0].rgb = filtered.rgb;
			//gl_FragData[0].rgb = filtered.rgb * albedo;
			//gl_FragData[1].rgb = filtered.rgb * albedo;

		}
	}

/* DRAWBUFFERS:3 */
}
