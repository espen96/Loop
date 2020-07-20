#version 120
#extension GL_EXT_gpu_shader4 : enable


//Prepares sky textures (2 * 256 * 256), computes light values and custom lightmaps
#include "/lib/settings.glsl"

flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;
flat varying vec3 lightSourceColor;
flat varying vec3 sunColor;
flat varying vec3 sunColorCloud;
flat varying vec3 moonColor;
flat varying vec3 moonColorCloud;
flat varying vec3 zenithColor;
flat varying vec3 avgSky;
flat varying vec2 tempOffsets;
flat varying float exposure;
flat varying float rodExposure;
flat varying float avgBrightness;
flat varying float exposureF;
flat varying float fogAmount;
flat varying float VFAmount;

uniform sampler2D colortex4;
uniform sampler2D noisetex;

uniform int frameCounter;
uniform float rainStrength;
uniform float eyeAltitude;
uniform vec3 sunVec;
uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
uniform float sunElevation;
uniform vec3 cameraPosition;
uniform float far;
uniform ivec2 eyeBrightnessSmooth;
#include "/lib/util.glsl"
#include "/lib/ROBOBO_sky.glsl"

float interleaved_gradientNoise(){
	vec2 coord = gl_FragCoord.xy;
	float noise = fract(52.9829189*fract(0.06711056*coord.x + 0.00583715*coord.y)+frameCounter/1.6180339887);
	return noise;
}
float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}
vec4 lightCol = vec4(lightSourceColor, float(sunElevation > 1e-5)*2-1.);
const float[17] Slightmap = float[17](14.0,17.,19.0,22.0,24.0,28.0,31.0,40.0,60.0,79.0,93.0,110.0,132.0,160.0,197.0,249.0,249.0);

void main() {
/* DRAWBUFFERS:4 */
gl_FragData[0] = vec4(0.0);
//Lightmap for forward shading (contains average integrated sky color across all faces + torch + min ambient)
vec3 avgAmbient = (ambientUp + ambientLeft + ambientRight + ambientB + ambientF + ambientDown)/6.;
if (gl_FragCoord.x < 17. && gl_FragCoord.y < 17.){
  float torchLut = clamp(16.0-gl_FragCoord.x,0.5,15.5);
  torchLut = torchLut+0.712;
  float torch_lightmap = max(1.0/torchLut/torchLut - 1/16.21/16.21,0.0);
  torch_lightmap = torch_lightmap*E_TORCH_AMOUNT*5.0;
  float sky_lightmap = 0.5*150.0;
  vec3 ambient = avgAmbient*sky_lightmap+torch_lightmap*vec3(E_TORCH_R,E_TORCH_G,E_TORCH_B)*E_TORCH_AMOUNT+E_MIN_LIGHT_AMOUNT*0.005/(exposureF+clamp(rodExposure*exposureF/10.,0.0,10000.0));
  gl_FragData[0] = vec4(ambient*E_Ambient_Mult,1.0);
}

//Lightmap for deferred shading (contains only torch + min ambient)
if (gl_FragCoord.x < 17. && gl_FragCoord.y > 19. && gl_FragCoord.y < 19.+17. ){
	float torchLut = clamp(16.0-gl_FragCoord.x,0.5,15.5);
  torchLut = torchLut+0.712;
  float torch_lightmap = max(1.0/torchLut/torchLut - 1/16.21/16.21,0.0);
  float ambient = torch_lightmap*E_TORCH_AMOUNT*5.;
  float sky_lightmap = 1.0;
  gl_FragData[0] = vec4(sky_lightmap,ambient,E_MIN_LIGHT_AMOUNT*0.005/(exposureF+clamp(rodExposure*exposureF/10.,0.0,10000.0)),1.0)*E_Ambient_Mult;
}

//Save light values
if (gl_FragCoord.x < 1. && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(ambientUp,1.0);
if (gl_FragCoord.x > 1. && gl_FragCoord.x < 2.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(ambientDown,1.0);
if (gl_FragCoord.x > 2. && gl_FragCoord.x < 3.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(ambientLeft,1.0);
if (gl_FragCoord.x > 3. && gl_FragCoord.x < 4.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(ambientRight,1.0);
if (gl_FragCoord.x > 4. && gl_FragCoord.x < 5.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(ambientB,1.0);
if (gl_FragCoord.x > 5. && gl_FragCoord.x < 6.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(ambientF,1.0);
if (gl_FragCoord.x > 6. && gl_FragCoord.x < 7.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(lightSourceColor,1.0);
if (gl_FragCoord.x > 7. && gl_FragCoord.x < 8.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(avgAmbient,1.0);

//Sky gradient (no clouds)
const float pi = 3.141592653589793238462643383279502884197169;
if (gl_FragCoord.x > 18. && gl_FragCoord.y > 1. && gl_FragCoord.x < 18+257){
  gl_FragData[0] = vec4(vec3(1.0,0.4,0.12)*100.,1.0);
}

//Temporally accumulate sky and light values
vec3 temp = texelFetch2D(colortex4,ivec2(gl_FragCoord.xy),0).rgb;
vec3 curr = gl_FragData[0].rgb*150.;
gl_FragData[0].rgb = clamp(mix(temp,curr,0.06),0.0,65000.);

//Exposure values
if (gl_FragCoord.x > 10. && gl_FragCoord.x < 11.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(exposure,avgBrightness,exposureF,1.0);
if (gl_FragCoord.x > 14. && gl_FragCoord.x < 15.  && gl_FragCoord.y > 19.+18. && gl_FragCoord.y < 19.+18.+1 )
gl_FragData[0] = vec4(rodExposure,0.0,0.0,1.0);

}
