#version 120

#include "/lib/settings.glsl"




//Horizontal bilateral blur for volumetric fog + Forward rendered objects + Draw volumetric fog
#extension GL_EXT_gpu_shader4 : enable



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
#include "/lib/waterBump.glsl"
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
  vec4 vl = vec4(0.0);
  float sum = 0.0;
  mat3x3 weights;
  ivec2 posD = ivec2(coord/2.0)*2;
  ivec2 posVl = ivec2(coord/2.0);
  float dz = zMults.x;
  ivec2 pos = (ivec2(gl_FragCoord.xy) % 2 )*2;

  ivec2 tcDepth =  posD + ivec2(-4,-4) + pos*2;
  float dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  float w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(-2)+pos,0)*w;
  sum += w;

	tcDepth =  posD + ivec2(-4,0) + pos*2;
  dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(-2,0)+pos,0)*w;
  sum += w;

	tcDepth =  posD + ivec2(0) + pos*2;
  dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(0)+pos,0)*w;
  sum += w;

	tcDepth =  posD + ivec2(0,-4) + pos*2;
  dsample = ld(texelFetch2D(depth,tcDepth,0).r);
  w = abs(dsample-frDepth) < dz ? 1.0 : 1e-5;
  vl += texelFetch2D(tex,posVl+ivec2(0,-2)+pos,0)*w;
  sum += w;

  return vl/sum;
}
	vec2 newtc = texcoord.xy;
	float GetDepthLinear(in vec2 coord) { //Function that retrieves the scene depth. 0 - 1, higher values meaning farther away
		return 1.0f * near * far / (far + near - (1.91f * texture2D(gdepthtex, coord).x - 1.0f) * (far - near));
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

  
  bool iswater = trpData.a > 0.9;
  bool isglass = (trpData.a > 0.2);
  
  



  vec2 refractedCoord = texcoord;
  vec2 refractedCoord2 = texcoord;

  




  
  if (iswater||isglass){

    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
  	vec3 np3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition;
    float norm = glassRefraction(np3.xz*1.71, 4.0, 0.25, 1.0,isglass);
    float displ = norm/(length(fragpos)/far)/80.;
    if(!isglass)refractedCoord += displ;
    if(isglass)refractedCoord += vec2(0.0,-displ);

    if (texture2D(colortex3,refractedCoord).a < 0.9 && !isglass)
     refractedCoord = texcoord;	
	
	


  }
  

  
  vec4 transparencies = texture2D(colortex2,refractedCoord);  
  
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

	
	
  }
	if(isEyeInWater == 2) {
		float depth = texture2D(depthtex0, newtc).r;
        color *= vl.a;
		vec3 fogColor = pow(vec3(255, 87, 0) / 255.0, vec3(2.5));

		color = mix(color, fogColor*10, min(GetDepthLinear(texcoord.st) * 590.0 / far, 1.0))*2;
	}
  
  
  color += vl.rgb;
  gl_FragData[0].r = vl.a;
  gl_FragData[1].rgb = clamp(color,6.11*1e-5,65000.0);
 //if(hand) gl_FragData[1].rgb = vec3(1);

}

