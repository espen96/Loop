#version 130
//Filter test

#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/encode.glsl"



uniform sampler2D colortex3;


flat varying float exposureA;
flat varying float tempOffsets;
uniform sampler2D colortex1;

uniform sampler2D colortex5;
uniform sampler2D colortex2;
uniform sampler2D colortex6;
uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D depthtex2;
uniform sampler2D noisetex;//depth
uniform int frameCounter;
flat varying vec2 TAA_Offset;
uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform float viewHeight;
uniform float viewWidth;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferPreviousModelView;
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
#include "/lib/res_params.glsl"
#include "/lib/projections.glsl"
uniform float far;
uniform float near;


vec2 texcoord = gl_FragCoord.xy*texelSize;	





//  smartDeNoise - parameters
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  sampler2D tex     - sampler image / texture
//  vec2 uv           - actual fragment coord
//  float sigma  >  0 - sigma Standard Deviation
//  float kSigma >= 0 - sigma coefficient 
//      kSigma * sigma  -->  radius of the circular kernel
//  float threshold   - edge sharpening threshold 
 
vec3 getDepthPoint(vec2 coord, float depth) {
    vec4 pos;
    pos.xy = coord;
    pos.z = depth;
    pos.w = 1.0;
    pos.xyz = pos.xyz * 2.0 - 1.0; //convert from the 0-1 range to the -1 to +1 range
    pos = gbufferProjectionInverse * pos;
    pos.xyz /= pos.w;
    
    return pos.xyz;
}
#define INV_SQRT_OF_2PI 0.39894228040143267793994605993439  // 1.0/SQRT_OF_2PI
#define INV_PI 0.31830988618379067153776752674503	
#define pow2(x) (x * x)
#define pow3(x) (pow2(x) * x)
#define pow4(x) (pow2(x) * pow2(x))
#define pow16(x) (pow4(x) * pow4(x))
#define pow32(x) (pow16(x) * pow16(x))
#define pow64(x) (pow32(x) * pow32(x))
#define pow128(x) (pow64(x) * pow64(x))	
#define clamp01(x) clamp(x, 0.0, 1.0)

#define max0(x) max(x, 0.0)	
	

vec3 atrous(vec2 coord, int pass) {
    int kernel = 1 << pass;
	float weights = 0.0;

    float origDepth = textureLod(depthtex0, coord, 0).r;
    float origDist = getDepthPoint(coord, origDepth).z;
	vec3 origNormal = (textureLod(colortex3, coord, 0).rgb);

    vec3 col = vec3(0);

	for (int i = -kernel; i <= kernel; i += 1 << pass) {
		for (int j = -kernel; j <= kernel; j += 1 << pass) {
			ivec2 icoord = ivec2(gl_FragCoord.xy) + ivec2(vec2(i,j));
			
			vec3 normal = (texelFetch(colortex3, icoord, 0).rgb);
			float depth = texelFetch(depthtex0, icoord, 0).r;
            float depthDist = getDepthPoint(vec2(icoord)*vec2(viewWidth, viewHeight), depth).z;
			vec3 color = texelFetch(colortex2, icoord, 0).rgb;
			
			float weight = 1.0;
            weight *= pow(length(16 - vec2(i,j)) / 16.0, 2.0);

			weight *= pow128(max0(dot(origNormal, normal)));


			weight *= clamp01(1.0-abs(origDist - depthDist));

            if(depth == 1.0) {
                weight = 0.0;
            }
			
			col += color * weight;
			weights += weight;
		}
	}
    col = col / weights;

    return col;
}

void main() {

/* DRAWBUFFERS:2 */



	float z = texture2D(depthtex1,texcoord).x;


#ifdef RT_FILTER
vec3 color = texture2D(colortex2,texcoord).rgb;

 if(z<1) color = atrous(texcoord, 1).rgb;
 
	gl_FragData[0].rgb =color; 
#else
	vec3 color2 = texture2D(colortex2,texcoord).rgb;
	gl_FragData[0].rgb = color2;
#endif




}
