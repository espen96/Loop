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
uniform sampler2D colortex7;
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


varying vec2 texcoord;
#define pow2(x) (x * x)
#define pow3(x) (pow2(x) * x)
#define pow4(x) (pow2(x) * pow2(x))
#define pow16(x) (pow4(x) * pow4(x))
#define pow32(x) (pow16(x) * pow16(x))
#define pow64(x) (pow32(x) * pow32(x))
#define pow128(x) (pow64(x) * pow64(x))	

float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}




//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 unenc,vec2 lightmap)
{    
	unenc.xy = unenc.xy / dot(abs(unenc), vec3(1.0)) + 0.00390625;
	unenc.xy = unenc.z <= 0.0 ? (1.0 - abs(unenc.yx)) * sign(unenc.xy) : unenc.xy;
    vec2 encn = unenc.xy * 0.5 + 0.5;
	
    return vec4((encn),vec2(lightmap.x,lightmap.y));
}

//encoding by jodie
float encodeVec2(vec2 a){
    const vec2 constant1 = vec2( 1., 256.) / 65535.;
    vec2 temp = floor( a * 254. );
	return temp.x*constant1.x+temp.y*constant1.y;
}
float encodeVec2(float x,float y){
    return encodeVec2(vec2(x,y));
}

#include "/lib/filter.glsl"



vec3 atrous2(vec2 coord, int pass) {
    int kernel = 1 << pass;
    ivec2 d_pos     = ivec2(coord * vec2(viewWidth, viewHeight));
    ivec2 c_pos     = ivec2(coord * vec2(viewWidth, viewHeight));	
	
	
	float weights = 0.0;

    float origDepth = textureLod(depthtex0, coord, 0).r;

	
	vec4 data2 = texture(colortex1, coord).rgba;
	vec4 dataUnpacked1 = vec4(decodeVec2(data2.x),decodeVec2(data2.y));			
	vec3 origNormal = decode(dataUnpacked1.yw);	

    vec3 col = vec3(0);

	for (int i = -kernel; i <= kernel; i += 1 << pass) {
		for (int j = -kernel; j <= kernel; j += 1 << pass) {
			ivec2 icoord = ivec2(gl_FragCoord.xy) + ivec2(vec2(i,j));
			

			vec4 data = (texelFetch(colortex1, icoord, 0).rgba);

			vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
			vec4 dataUnpacked2 = vec4(decodeVec2(data.z),decodeVec2(data.w));
			
			vec3 normal = decode(dataUnpacked0.yw);
			
			float depth = texelFetch(depthtex0, icoord, 0).r;

			vec3 color = vec3(dataUnpacked2.yz,0);
			
			float weight = 1.0;
            weight *= pow(length(16 - vec2(i,j)) / 16.0, 2.0);

			weight *= pow128(clamp(dot(origNormal, normal),0,1));


	

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

/* DRAWBUFFERS:7 */



		vec4 data = texture2D(colortex1,texcoord);
		vec4 mask = vec4(texture2D(colortex7,texcoord).rgba);
		float mask2 = luma(mask.rgb);

		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));	
		vec2 lightmap = vec2(dataUnpacked1.yz);	
	//	vec2 lightmap = clamp(atrous2(texcoord, 0).rg,0,1);	
		vec3 albedo = (vec3(dataUnpacked0.xz,dataUnpacked1.x));
		vec3 normal = decode(dataUnpacked0.yw);


	float z = texture2D(depthtex1,texcoord).x;

	vec4 data1 = encode(normal,lightmap);
	vec4 data0 = vec4(albedo,dataUnpacked1.w);

vec4 color = texture2D(colortex5,texcoord);



if(z <1 &&mask2 > 0.001) color.rgb = atrous3(texcoord,2).rgb;


	gl_FragData[0] = color; 






}