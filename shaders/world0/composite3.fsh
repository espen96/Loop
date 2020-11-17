#version 120

#include "/lib/settings.glsl"




//Horizontal bilateral blur for volumetric fog + Forward rendered objects + Draw volumetric fog
#extension GL_EXT_gpu_shader4 : enable


uniform float blindness; 
varying vec2 texcoord;
flat varying vec3 zMults;
uniform sampler2D depthtex0;
uniform sampler2D colortex1;
uniform sampler2D colortex3;
uniform sampler2D colortex2;
uniform sampler2D colortex0;
uniform sampler2D noisetex;
uniform sampler2D gdepthtex;
uniform float frameTimeCounter;

uniform int frameCounter;
uniform float far;
uniform float near;
uniform int isEyeInWater;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform vec2 texelSize;
uniform vec3 cameraPosition;

#include "/lib/waterOptions.glsl"
#include "/lib/encode.glsl"
#include "/lib/res_params.glsl"

float ld(float depth) {
    return 1.0 / (zMults.y - depth * zMults.z);		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
vec4 BilateralUpscale(sampler2D tex, sampler2D depth,vec2 coord,float frDepth){
  coord = coord;
  vec4 vl = vec4(0.0);
  float sum = 0.0;
  mat3x3 weights;
  const ivec2 scaling = ivec2(1.0/VL_RENDER_RESOLUTION);
  ivec2 posD = ivec2(coord*VL_RENDER_RESOLUTION)*scaling;
  ivec2 posVl = ivec2(coord*VL_RENDER_RESOLUTION);
  float dz = zMults.x;
  ivec2 pos = (ivec2(gl_FragCoord.xy+frameCounter) % 2 )*2;
	//pos = ivec2(1,-1);

  ivec2 tcDepth =  posD + ivec2(-2,-2) * scaling + pos * scaling;
  float dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  float w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(-2)+pos,0)*w;
  sum += w;

	tcDepth =  posD + ivec2(-2,0) * scaling + pos * scaling;
  dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(-2,0)+pos,0)*w;
  sum += w;

	tcDepth =  posD + ivec2(0) + pos * scaling;
  dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(0)+pos,0)*w;
  sum += w;

	tcDepth =  posD + ivec2(0,-2) * scaling + pos * scaling;
  dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(0,-2)+pos,0)*w;
  sum += w;

  return vl/sum;
}


	
	

float getWaterHeightmap(vec2 posxz, float iswater) {
	vec2 pos = posxz;
  float moving = clamp(iswater*2.-1.0,0.0,1.0);
	vec2 movement = vec2(-0.005*frameTimeCounter*moving,0.0);
	float caustic = 0.0;
	float weightSum = 0.0;
	float radiance =  2.39996;
	mat2 rotationMatrix  = mat2(vec2(cos(radiance),  -sin(radiance)),  vec2(sin(radiance),  cos(radiance)));
	for (int i = 1; i < 3; i++){
		vec2 displ = texture2D(noisetex, pos/32.0/1.74/1.74 + movement).bb*2.0-1.0;
    float wave = texture2D(noisetex, (pos*vec2(3., 1.0)/128. + movement + displ/128.0)*exp(i*1.0)).b;
		caustic += wave*exp(-i*1.0);
		weightSum += exp(-i*1.0);
		pos = rotationMatrix * pos;
	}
	return caustic / weightSum;
}	
	
	
	
	
	
	
void main() {
										
  /* DRAWBUFFERS:23 */
 	vec2 texcoord = gl_FragCoord.xy*texelSize;							 
	float masks = texture2D(colortex3,texcoord).a;
	gl_FragData[1].a = masks;		 
		vec4 data = texture2D(colortex1,texcoord);

		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
  //3x3 bilateral upscale from half resolution
  float z = texture2D(depthtex0,texcoord).x;
  float frDepth = ld(z);
  vec4 vl = BilateralUpscale(colortex0,depthtex0,gl_FragCoord.xy,frDepth);


  bool hand = abs(frDepth) <0.01;

  vec4 trpData = texture2D(colortex3,texcoord);
vec3 mask2 =vec3(0,0,0);
  

  
  


  

  

if(trpData.a > 0.2 && trpData.a <0.9) mask2=vec3(0,0,1);
if(trpData.a >0.9 && trpData.a<1.1) mask2=vec3(0,0,1);

if(trpData.a >0.90 && trpData.a<0.902) mask2=vec3(1,0,0);
if(trpData.a >0.901 && trpData.a<0.99) mask2=vec3(0,0,1);
  bool isglass = (mask2.b > 0);

bool  iswater = (mask2.r  > 0); 

//if(trpData.a >0.901) trpData.a = 0.1;

  
//  isglass = (trpData.a > 1);





  vec2 refractedCoord = texcoord;
  vec2 refractedCoord2 = texcoord;

  




  
  if (iswater){
    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
  	vec3 np3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition;
    float norm = getWaterHeightmap(np3.xz+np3.y, 1.0)-0.5;
    float displ = norm/(length(fragpos)/far)/2000. * (1.0 + isEyeInWater*2.0);
    refractedCoord += displ*RENDER_SCALE;

    if (texture2D(colortex3,refractedCoord).a < 0.90)
      refractedCoord = texcoord;

  }
    refractedCoord2 = refractedCoord;
  
	vec4 transparencies = texture2D(colortex2,refractedCoord);  
	if(hand)transparencies = texture2D(colortex2,texcoord);  
	  
	  
	  
	  
	  
	  
  vec3 color = texture2D(colortex3,refractedCoord).rgb;
  

//  if (frDepth > 2.5/far || transparencies.a < 0.99)  // Discount fix for transparencies through hand
    color = color*(1.0-transparencies.a)+transparencies.rgb*10.;

  float dirtAmount = Dirt_Amount;
	vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
	vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
	vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;

  color *= vl.a;
  if (isEyeInWater == 1){
    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
    color.rgb *= exp(-length(fragpos)*totEpsilon);
    vl.a *= dot(exp(-length(fragpos)*totEpsilon),vec3(0.2,0.7,0.1))*0.5+0.5;
	
	
  }
  if (isEyeInWater == 2){
    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
    color.rgb *= exp(-length(fragpos)*vec3(0.2,0.7,4.0)*4.);
    color.rgb += vec3(4.0,0.31,0.1)*2;
    vl.a = 0.0;
  }
  else
    color += vl.rgb;
  color += vl.rgb;
  gl_FragData[0].r = vl.a;
  gl_FragData[1].rgb = clamp(color,6.11*1e-5,65000.0);
 //gl_FragData[1].rgb = vec3(trpData.a);
  // gl_FragData[1].rgb += vec3(mask2);

}

