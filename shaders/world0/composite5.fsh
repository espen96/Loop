#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D colortex3;
uniform sampler2D depthtex1;
uniform sampler2D noisetex;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform float viewWidth;
uniform float viewHeight;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform vec2 texelSize;
uniform int frameCounter;

vec2 texcoord = gl_FragCoord.xy*texelSize;
#include "/lib/res_params.glsl"
#include "/lib/settings.glsl"



float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}

//////////////////////////////////////////////////////////////////////////////////////////////////



vec3 MotionBlur(vec3 color, float z, float dither){
	
	float hand = float(z < 0.56);

	if (hand < 0.5){
		float mbwg = 0.0;
		vec2 doublePixel = 2.0 / vec2(viewWidth, viewHeight);
		vec3 mblur = vec3(0.0);
		
		vec4 currentPosition = vec4(texcoord, z, 1.0) * 2.0 - 1.0;
		
		vec4 viewPos = gbufferProjectionInverse * currentPosition;
		viewPos = gbufferModelViewInverse * viewPos;
		viewPos /= viewPos.w;
		
		vec3 cameraOffset = cameraPosition - previousCameraPosition;
		
		vec4 previousPosition = viewPos + vec4(cameraOffset, 0.0);
		previousPosition = gbufferPreviousModelView * previousPosition;
		previousPosition = gbufferPreviousProjection * previousPosition;
		previousPosition /= previousPosition.w;

		vec2 velocity = (currentPosition - previousPosition).xy;
		velocity = velocity / (1.0 + length(velocity)) * MOTION_BLUR_STRENGTH * 0.02;
		
		vec2 coord = texcoord.st - velocity * (3.5 + dither);
		for(int i = 0; i < 9; i++, coord += velocity){
			vec2 coordb = clamp(coord, doublePixel, 1.0 - doublePixel);
			mblur += texture2DLod(colortex3, coordb, 0.0).rgb;
			mbwg += 1.0;
		}
		mblur /= mbwg;

		return mblur;
	}
	else return color;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

/* DRAWBUFFERS:3 */


void main() {	

	float noise = blueNoise();
	float z = texture2D(depthtex1, texcoord.xy).x;
	vec3 color = texture2D(colortex3, texcoord.xy).xyz;



	#ifdef MOTION_BLUR
		color = MotionBlur(color, z, noise);
	#endif

	gl_FragData[0] = vec4(color, 1.0);
}
