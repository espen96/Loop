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




#define DENOISE_RANGE1 vec2(32, 30)


#include "/lib/filter.glsl"
void main() {

/* DRAWBUFFERS:8 */




	float z = texture2D(depthtex1,texcoord).x;

	vec4 trpData = texture2D(colortex7,texcoord);
	bool iswater = texture2D(colortex7,texcoord).a > 0.99;
	vec3	 color = texture2D(colortex8,texcoord).rgb;		
	
#ifdef ssptfilter
#ifdef filterpass_0
	if (z <1 && !iswater) color.rgb = clamp(atrous3(texcoord.xy*RENDER_SCALE,24,colortex8,0.0).rgb,0,10);


#endif
#endif





	gl_FragData[0].rgb = color;





	

}
