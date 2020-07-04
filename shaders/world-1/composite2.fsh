#version 120
//Temporal sspt filter

#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/encode.glsl"





//no need to clear the buffers, saves a few fps
const bool colortex0Clear = false;
const bool colortex1Clear = false;
const bool colortex2Clear = true;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
const bool colortex7Clear = false;

varying vec2 texcoord;
flat varying float exposureA;
flat varying float tempOffsets;
uniform sampler2D colortex1;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D depthtex0;

uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform float viewHeight;
uniform float viewWidth;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferPreviousModelView;
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
#include "/lib/projections.glsl"


		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		vec4 entityg = texture2D(colortex7,texcoord);
		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool emissive = abs(dataUnpacked1.w-0.9) >0.4;
		vec3 filtered = texture2D(colortex3,texcoord).rgb;
		vec3 test = texture2D(colortex5,texcoord).rgb;
		bool entity = abs(entityg.r) >0.9;




//approximation from SMAA presentation from siggraph 2016


vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}


vec3 TAA_sspt(){
	//use velocity from the nearest texel from camera in a 3x3 box in order to improve edge quality in motion


	vec3 closestToCamera = vec3(texcoord,texture2D(depthtex0,texcoord).x);

	//reproject previous frame
	vec3 fragposition = toScreenSpace(closestToCamera);
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;
	previousPosition.xy = texcoord + velocity;


	//reject history if off-screen and early exit
	//Samples current frame 3x3 neighboorhood
	vec3 albedoCurrent0 = texture2D(colortex3, texcoord).rgb;


	vec3 albedoPrev = texture2D(colortex5, previousPosition.xy).xyz;
	vec3 supersampled =  mix(albedoPrev,albedoCurrent0,clamp(0.75,0.0,1.0));
	if (hand|| entity ||emissive) supersampled = albedoCurrent0;





	//De-tonemap
	return supersampled;
}

void main() {

/* DRAWBUFFERS:3 */
#ifdef SSPT
	vec3 color = TAA_sspt();
	if(entity) color = texture2D(colortex3,texcoord).rgb;
	gl_FragData[0].rgb = color;
#else
	vec3 color2 = texture2D(colortex3,texcoord).rgb;
	gl_FragData[0].rgb = color2;
#endif





}
