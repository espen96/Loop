#version 130
//Filter test

#extension GL_EXT_gpu_shader4 : enable






uniform sampler2D colortex1;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex7;


uniform sampler2D colortexA;
uniform sampler2D colortexC;
uniform sampler2D colortexD;
uniform sampler2D colortexE;
uniform sampler2D colortex8;
uniform sampler2D colortex9;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;




uniform mat4 gbufferPreviousModelView;

uniform sampler2D noisetex;//depth
uniform int frameCounter;
flat varying vec2 TAA_Offset;
uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform float viewHeight;
uniform float viewWidth;
uniform vec3 previousCameraPosition;

#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
#define denoise
#define power
#include "/lib/res_params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/projections.glsl"

uniform float far;
uniform float near;
uniform float aspectRatio;

vec2 texcoord = gl_FragCoord.xy*texelSize;	


vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}



float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
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


#define DENOISE_RANGE1 vec2(8, 8)





#include "/lib/filter.glsl"




void main() {

/* DRAWBUFFERS:8 */

	vec3	 color = texture2D(colortex8,texcoord).rgb;		
		
		
float z = texture2D(depthtex1,texcoord).x;







#ifdef ssptfilter
#ifdef filerpass_2
	if (z <1) color.rgb = clamp(atrous3(texcoord.xy*RENDER_SCALE,6,colortex8,0.0).rgb,0,10);


#endif
#endif


	gl_FragData[0].rgb = color;


	

}
