#version 120


#include "/lib/settings.glsl"
#include "/lib/res_params.glsl"




#extension GL_EXT_gpu_shader4 : enable



flat varying vec4 lightCol;
flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;
flat varying float tempOffsets;
flat varying float fogAmount;
flat varying float VFAmount;

uniform sampler2D colortex4;
uniform float sunElevation;
uniform float rainStrength;
uniform int isEyeInWater;
uniform int frameCounter;
uniform int worldTime;
#include "/lib/util.glsl"
float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
	tempOffsets = HaltonSeq2(frameCounter%10000);
	gl_Position = ftransform();
	gl_Position.xy = (gl_Position.xy*0.5+0.5)*0.51*2.0-1.0;
	#ifdef TAA_UPSCALING
		gl_Position.xy = (gl_Position.xy*0.5+0.5)*RENDER_SCALE*2.0-1.0;
	#endif
	vec3 sc = texelFetch2D(colortex4,ivec2(6,37),0).rgb;
	vec3 avgAmbient = texelFetch2D(colortex4,ivec2(11,37),0).rgb;
	ambientUp = texelFetch2D(colortex4,ivec2(0,37),0).rgb;
	ambientDown = texelFetch2D(colortex4,ivec2(1,37),0).rgb;
	ambientLeft = texelFetch2D(colortex4,ivec2(2,37),0).rgb;
	ambientRight = texelFetch2D(colortex4,ivec2(3,37),0).rgb;
	ambientB = texelFetch2D(colortex4,ivec2(4,37),0).rgb;
	ambientF = texelFetch2D(colortex4,ivec2(5,37),0).rgb;


	lightCol.a = float(sunElevation > 1e-5)*2-1.;
	lightCol.rgb = sc;
	lightCol.rgb *= (1.0-rainStrength*0.9);

	float modWT = (worldTime%24000)*1.0;

	float fogAmount0 = 1/3000.+FOG_TOD_MULTIPLIER*(1/180.*(clamp(modWT-11000.,0.,2000.0)/2000.+(1.0-clamp(modWT,0.,3000.0)/3000.))*(clamp(modWT-11000.,0.,2000.0)/2000.+(1.0-clamp(modWT,0.,3000.0)/3000.)) + 1/200.*clamp(modWT-13000.,0.,1000.0)/1000.*(1.0-clamp(modWT-23000.,0.,1000.0)/1000.));
	VFAmount = CLOUDY_FOG_AMOUNT*(fogAmount0*fogAmount0+FOG_RAIN_MULTIPLIER*1.8/20000.*rainStrength);
	fogAmount = BASE_FOG_AMOUNT*(fogAmount0+max(FOG_RAIN_MULTIPLIER*1/10.*rainStrength , FOG_TOD_MULTIPLIER*1/50.*clamp(modWT-13000.,0.,1000.0)/1000.*(1.0-clamp(modWT-23000.,0.,1000.0)/1000.)));

}