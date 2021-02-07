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



vec3 closestToCamera5taps(vec2 texcoord)
{
	vec2 du = vec2(texelSize.x*2., 0.0);
	vec2 dv = vec2(0.0, texelSize.y*2.);

	vec3 dtl = vec3(texcoord,0.) + vec3(-texelSize, texture2D(depthtex0, texcoord - dv - du).x);
	vec3 dtr = vec3(texcoord,0.) +  vec3( texelSize.x, -texelSize.y, texture2D(depthtex0, texcoord - dv + du).x);
	vec3 dmc = vec3(texcoord,0.) + vec3( 0.0, 0.0, texture2D(depthtex0, texcoord).x);
	vec3 dbl = vec3(texcoord,0.) + vec3(-texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv - du).x);
	vec3 dbr = vec3(texcoord,0.) + vec3( texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv + du).x);

	vec3 dmin = dmc;
	dmin = dmin.z > dtr.z? dtr : dmin;
	dmin = dmin.z > dtl.z? dtl : dmin;
	dmin = dmin.z > dbl.z? dbl : dmin;
	dmin = dmin.z > dbr.z? dbr : dmin;
	#ifdef TAA_UPSCALING
	dmin.xy = dmin.xy/RENDER_SCALE;
	#endif
	return dmin;
}


#define DENOISE_RANGE1 vec2(32, 30)

#include "/lib/filter.glsl"



#define comp(a, b, c) (v[a].c > v[b].c)
#define swap(a, b, c) temp = v[b].c; v[b].c = v[a].c; v[a].c = temp;
#define cmpswap(x, y) if (comp(x, y, r)){ swap(x, y, r)}; if (comp(x, y, g)){ swap(x, y, g)}; if (comp(x, y, b)){ swap(x, y, b)};

vec3 medianSub(in vec3[9] v){
    float temp;  
        
    cmpswap(0, 1); 
    cmpswap(3, 4); 
    cmpswap(6, 7); 
    
    cmpswap(1, 2); 
    cmpswap(4, 5); 
    cmpswap(7, 8); 
    
    cmpswap(0, 1); 
    cmpswap(3, 4); 
    cmpswap(6, 7); 
    cmpswap(2, 5); 
    
    cmpswap(0, 3); 
    cmpswap(1, 4); 
    cmpswap(5, 8); 
    
    cmpswap(3, 6); 
    cmpswap(4, 7); 
    cmpswap(2, 5); 
    
    cmpswap(0, 3); 
    cmpswap(1, 4); 
    cmpswap(5, 7); 
    cmpswap(2, 6); 
    
    cmpswap(1, 3); 
    cmpswap(4, 6); 
    
    cmpswap(2, 4); 
    cmpswap(5, 6); 
    
    cmpswap(2, 3); 
    
    return (v[4]);
}


vec3 median(sampler2D tex1,vec2 uv){
	vec2 pixel = 1. / vec2(viewWidth,viewHeight).xy;
    vec4 o = vec4(pixel.x, 0., pixel.y, -pixel.y); 
    vec3 n[9];
    n[0] = texture(tex1, uv - o.xz).rgb;  
    n[1] = texture(tex1, uv - o.yz).rgb;
    n[2] = texture(tex1, uv + o.xw).rgb;
    n[3] = texture(tex1, uv - o.xy).rgb;
    n[4] = texture(tex1, uv + o.xy).rgb;
    n[5] = texture(tex1, uv - o.xw).rgb;
    n[6] = texture(tex1, uv + o.yz).rgb;
    n[7] = texture(tex1, uv + o.xz).rgb;     
    n[8] = texture(tex1, uv).rgb;  
    return medianSub(n);
}


void main() {

/* DRAWBUFFERS:8E */

	vec3	 color = texture2D(colortex8,texcoord).rgb;		
		
		
float z = texture2D(depthtex1,texcoord).x;




    vec2 variance  = computeVariance(colortex5, ivec2(floor(texcoord * vec2(viewWidth, viewHeight)/RENDER_SCALE)));	


#ifdef ssptfilter
	if (z <1) color.rgb = clamp(atrous3(texcoord.xy*RENDER_SCALE,24,colortex8,0.0).rgb,0,10);


#endif


	gl_FragData[0].rgb = color;

	gl_FragData[1].rgb = mix(median(colortexE,gl_FragCoord.xy/vec2(viewWidth,viewHeight).xy),texture2D(colortexE,texcoord).rgb,0.75);



	

}
