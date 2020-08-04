#version 120
//sspt filter2
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"

const float eyeBrightnessHalflife = 5.0f;





uniform sampler2D colortex0;//clouds
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex6;//Skybox
uniform sampler2D depthtex2;//depth
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth
uniform sampler2D texture;





uniform int frameCounter;
uniform float frameTime;

uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjectionInverse;
uniform float far;
uniform float near;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferModelView;


uniform vec2 texelSize;
uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;
uniform vec3 cameraPosition;
uniform int framemod8;
uniform vec3 sunVec;
vec2 texcoord = gl_FragCoord.xy*texelSize;	
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)


vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}



#include "/lib/color_transforms.glsl"
#include "/lib/encode.glsl"




vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}

#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
float triangularize(float dither)
{
    float center = dither*2.0-1.0;
    dither = center*inversesqrt(abs(center));
    return clamp(dither-fsign(center),0.0,1.0);
}

vec3 fp10Dither(vec3 color,float dither){
	const vec3 mantissaBits = vec3(6.,6.,5.);
	vec3 exponent = floor(log2(color));
	return color + dither*exp2(-mantissaBits)*exp2(exponent);
}




float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}


#include "/lib/blur.glsl"



vec2 noise(vec2 coord)
{
     float x = sin(coord.x * 100.0) * 0.1 + sin((coord.x * 200.0) + 3.0) * 0.05 + fract(cos((coord.x * 19.0) + 1.0) * 33.33) * 0.15;
     float y = sin(coord.y * 100.0) * 0.1 + sin((coord.y * 200.0) + 3.0) * 0.05 + fract(cos((coord.y * 19.0) + 1.0) * 33.33) * 0.25;
	 return vec2(x,y);
}




void main() {

float z = texture2D(depthtex0,texcoord).x;



    vec2 p_m = texcoord;
    vec2 p_d = p_m;
    p_d.xy -= frameTimeCounter * 0.1;
    vec2 dst_map_val = vec2(noise(p_d.xy));
    vec2 dst_offset = dst_map_val.xy;

    dst_offset *= 2.0;

    dst_offset *= 0.01;
	
    //reduce effect towards Y top
	
    dst_offset *= (1. - p_m.t);	
    vec2 dist_tex_coord = p_m.st + (dst_offset*linZ(z)/3);

	vec2 coord = dist_tex_coord;

vec3 filtered = texture2D(colortex3,coord).rgb;
		 {
float Depth = texture2D(depthtex0, coord).x;

vec3 blur = texture2D(colortex3, coord).xyz;

#ifndef RT_FILTER

blur1 = filtered.rgb;

#else





	if (Depth < 1.0){
	Depth = ld(Depth);
    


	blur = ssaoVL_blur(coord,vec2(0.0,1.0),Depth*far);


}

#endif		    
		    gl_FragData[0].rgb = filtered.rgb;	
		
		   #ifdef SSPT
		    gl_FragData[0] = vec4((blur.xyz),1.0);
			#endif
			


		}
	

/* DRAWBUFFERS:3 */
}