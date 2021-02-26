
uniform mat4 gbufferModelView;
flat varying vec3 zMults;
uniform sampler2D depthtex0;
uniform sampler2D colortex7;
uniform sampler2D colortex3;
uniform sampler2D colortex4;
uniform sampler2D colortex10;
uniform sampler2D colortex11;
uniform sampler2D colortex12;
uniform sampler2D colortex13;
uniform sampler2D colortex2;
uniform sampler2D colortex0;
uniform sampler2D noisetex;
uniform float blindness;  
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
#include "/lib/res_params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/sky_gradient.glsl"
#define clamp01(x) clamp(x, 0.0, 1.0)
#define fsign(x) (clamp01(x * 1e35) * 2.0 - 1.0)
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
vec2 noise(vec2 coord)
{
     float x = sin(coord.x * 100.0) * 0.1 + sin((coord.x * 200.0) + 3.0) * 0.05 + fract(cos((coord.x * 19.0) + 1.0) * 33.33) * 0.15;
     float y = sin(coord.y * 100.0) * 0.1 + sin((coord.y * 200.0) + 3.0) * 0.05 + fract(cos((coord.y * 19.0) + 1.0) * 33.33) * 0.25;
	 return vec2(x,y);
}

vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos;

    return pos.xyz;
}

vec3 viewToWorld(vec3 viewPos) {

    vec4 pos;
    pos.xyz = viewPos;
    pos.w = 0.0;
    pos = gbufferModelViewInverse * pos;

    return pos.xyz;
}

void main() {
  vec2 texcoord = gl_FragCoord.xy*texelSize;
  vec2 texcoord2 = gl_FragCoord.xy*texelSize;
  vec2 texcoord3 = gl_FragCoord.xy*texelSize;
/* RENDERTARGETS: 7,3 */
  vec4 trpData = texture2D(colortex7,texcoord);
  bool iswater = trpData.a > 0.99;
  //3x3 bilateral upscale from half resolution
  float z = texture2D(depthtex0,texcoord).x;
  float frDepth = ld(z);
  vec4 vl = BilateralUpscale(colortex0,depthtex0,gl_FragCoord.xy,frDepth);
  bool istransparent = (texture2D(colortex2,texcoord).a) > 0.0;	


  vec4 normal2 = (texture2D(colortex10, texcoord));
  vec4 normal3 = (texture2D(colortex11, texcoord));
  vec3 worldnormal = vec3(normal3.r); 

	    float sigma = 0.25;
	    float intensity = exp(-sigma * texture2D(colortex2,texcoord).a);  
  gl_FragData[2].rgb = vec3( worldnormal);  		
		
   vec2 refractedCoord = texcoord; 
     #ifdef NETHER 
        vec2 p_m = texcoord;
    vec2 p_d = p_m;
    p_d.xy -= frameTimeCounter * 0.1;
    vec2 dst_map_val = vec2(noise(p_d.xy));
    vec2 dst_offset = dst_map_val.xy;

    dst_offset *= 2.0;

    dst_offset *= 0.01;
	
    //reduce effect towards Y top
	
    dst_offset *= (1. - p_m.t);	
    vec2 dist_tex_coord = p_m.st + (dst_offset*ld(z)*0.02);

	  vec2 coord = dist_tex_coord;

  refractedCoord = coord; 
  #endif
  float refraction = (1*clamp(1-abs(0 + (ld(z) - 0.0) * (1 - 0) / (1.0 - 0.0)),0,1)*0.005);
  float refraction2 = pow(texture2D(colortex2,texcoord).a,3)*2-0.2;

// if(istransparent || iswater)   texcoord.xy=texcoord.xy+worldnormal.xy;
  vec4 transparencies = texture2D(colortex2,texcoord);  


    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
  if (iswater){
    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
  	vec3 np3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition;
    float norm = getWaterHeightmap(np3.xz+np3.y, 1.0)-0.5;
    float displ = norm/(length(fragpos)/far)/2000. * (1.0 + isEyeInWater*2.0);
    refractedCoord += displ*RENDER_SCALE;

    if (texture2D(colortex7,refractedCoord).a < 0.99)
      refractedCoord = texcoord;

  }

// if(istransparent || iswater)   refractedCoord.xy=refractedCoord.xy+worldnormal.xy;

  vec3 color = texture2D(colortex3,refractedCoord).rgb;


	

  if (frDepth > 2.5/far || transparencies.a < 0.99)  // Discount fix for transparencies through hand
    color = color*(1.0-transparencies.a)+transparencies.rgb*15.;
	color.rgb = intensity * color.rgb;	

  float dirtAmount = Dirt_Amount;
	vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
	vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
	vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;

  color *= vl.a;

  if (isEyeInWater == 1){
    vec3 fragpos = toScreenSpace(vec3(texcoord,z));
    color.rgb *= exp(-length(fragpos)*totEpsilon);

    vl.a *= dot(exp(-length(fragpos)*totEpsilon),vec3(0.2,0.7,0.1))*0.5+0.5;
  }
  if (isEyeInWater == 2){
    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
    color.rgb *= exp(-length(fragpos)*vec3(0.2,0.7,4.0)*4.);
   
    color.rgb += vec3(4.0,0.5,0.1)*0.5;

    vl.a = 0.0;
  }
  if (blindness > 0){
    vec3 fragpos = toScreenSpace(vec3(texcoord-vec2(0.0)*texelSize*0.5,z));
    color.rgb *= exp(-length(fragpos)*vec3(1.0)*blindness);


    vl.a = 0.0;
  }  
  else
  
    color += vl.rgb;
	

	
	float fogFactorAbs = 1.0 - clamp((length(fragpos) - gl_Fog.start) * gl_Fog.scale, 0.0, 1.0);
   fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(0.0)*texelSize*0.5,z));	
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normalize(p3);

    vec3 fogColor = mix(skyCloudsFromTex(np3, colortex4).rgb, vec3(2.0), fogFactorAbs)*8/3./150.0;
//	if(z <1.0)   color.rgb = mix(fogColor, color, fogFactorAbs);
  

  gl_FragData[0].r = vl.a;
  gl_FragData[0].a = trpData.a;
  gl_FragData[1].rgb = clamp(color,6.11*1e-5,65000.0);



}
