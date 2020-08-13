#version 120
//Temporal sspt filter

#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/encode.glsl"



#define SSPT_FLICKER_REDUCTION 1.0
#define SSPT_BLEND_FACTOR 0.2



const int noiseTextureResolution = 32;


/*
const int colortex0Format = RGBA16F;				// low res clouds (deferred->composite2) + low res VL (composite5->composite15)
const int colortex1Format = RGBA16;					//terrain gbuffer (gbuffer->composite2)
const int colortex2Format = RGBA16F;				//forward + transparencies (gbuffer->composite4)
const int colortex3Format = RGBA16F;				//frame buffer + bloom (deferred6->final)
const int colortex4Format = RGBA16F;				//light values and skyboxes (everything)
const int colortex5Format = R11F_G11F_B10F;			//TAA buffer (everything)
const int colortex6Format = R11F_G11F_B10F;			//additionnal buffer for bloom (composite3->final)
const int colortex7Format = R11F_G11F_B10F;					//Final output, transparencies id (gbuffer->composite4)
*/
//no need to clear the buffers, saves a few fps
const bool colortex0Clear = false;
const bool colortex1Clear = false;
const bool colortex2Clear = true;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
const bool colortex7Clear = false;


flat varying float exposureA;
flat varying float tempOffsets;
uniform sampler2D colortex1;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
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

vec2 texcoord = gl_FragCoord.xy*texelSize;	

float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+tempOffsets*2);
}
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


		float z = texture2D(depthtex1,texcoord).x;
		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		vec3 filtered = texture2D(colortex3,texcoord).rgb;
		vec3 test = texture2D(colortex5,texcoord).rgb;

		bool issky = z >=1.0;
		bool iswater = texture2D(colortex3,texcoord).a > 0.9;


		float noise = blueNoise();

//returns the projected coordinates of the closest point to the camera in the 3x3 neighborhood
vec3 closestToCamera3x3()
{
	vec2 du = vec2(texelSize.x, 0.0);
	vec2 dv = vec2(0.0, texelSize.y);

	vec3 dtl = vec3(texcoord,0.) + vec3(-texelSize, texture2D(depthtex0, texcoord - dv - du).x);
	vec3 dtc = vec3(texcoord,0.) + vec3( 0.0, -texelSize.y, texture2D(depthtex0, texcoord - dv).x);
	vec3 dtr = vec3(texcoord,0.) +  vec3( texelSize.x, -texelSize.y, texture2D(depthtex0, texcoord - dv + du).x);

	vec3 dml = vec3(texcoord,0.) +  vec3(-texelSize.x, 0.0, texture2D(depthtex0, texcoord - du).x);
	vec3 dmc = vec3(texcoord,0.) + vec3( 0.0, 0.0, texture2D(depthtex0, texcoord).x);
	vec3 dmr = vec3(texcoord,0.) + vec3( texelSize.x, 0.0, texture2D(depthtex0, texcoord + du).x);

	vec3 dbl = vec3(texcoord,0.) + vec3(-texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv - du).x);
	vec3 dbc = vec3(texcoord,0.) + vec3( 0.0, texelSize.y, texture2D(depthtex0, texcoord + dv).x);
	vec3 dbr = vec3(texcoord,0.) + vec3( texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv + du).x);

	vec3 dmin = dmc;

	dmin = dmin.z > dtc.z? dtc : dmin;
	dmin = dmin.z > dtr.z? dtr : dmin;

	dmin = dmin.z > dml.z? dml : dmin;
	dmin = dmin.z > dtl.z? dtl : dmin;
	dmin = dmin.z > dmr.z? dmr : dmin;

	dmin = dmin.z > dbl.z? dbl : dmin;
	dmin = dmin.z > dbc.z? dbc : dmin;
	dmin = dmin.z > dbr.z? dbr : dmin;

	return dmin;
}
	vec2 tempOffset=TAA_Offset;

//Due to low sample count we "tonemap" the inputs to preserve colors and smoother edges
vec3 weightedSample(sampler2D colorTex, vec2 texcoord){
	vec3 wsample = texture2D(colorTex,texcoord).rgb*exposureA;
	return wsample/(1.0+luma(wsample));

}



//approximation from SMAA presentation from siggraph 2016
vec3 FastCatmulRom(sampler2D colorTex, vec2 texcoord, vec4 rtMetrics, float sharpenAmount)
{
    vec2 position = rtMetrics.zw * texcoord;
    vec2 centerPosition = floor(position - 0.5) + 0.5;
    vec2 f = position - centerPosition;
    vec2 f2 = f * f;
    vec2 f3 = f * f2;

    float c = sharpenAmount;
    vec2 w0 =        -c  * f3 +  2.0 * c         * f2 - c * f;
    vec2 w1 =  (2.0 - c) * f3 - (3.0 - c)        * f2         + 1.0;
    vec2 w2 = -(2.0 - c) * f3 + (3.0 -  2.0 * c) * f2 + c * f;
    vec2 w3 =         c  * f3 -                c * f2;

    vec2 w12 = w1 + w2;
    vec2 tc12 = rtMetrics.xy * (centerPosition + w2 / w12);
    vec3 centerColor = texture2D(colorTex, vec2(tc12.x, tc12.y)).rgb;

    vec2 tc0 = rtMetrics.xy * (centerPosition - 1.0);
    vec2 tc3 = rtMetrics.xy * (centerPosition + 2.0);
    vec4 color = vec4(texture2D(colorTex, vec2(tc12.x, tc0.y )).rgb, 1.0) * (w12.x * w0.y ) +
                   vec4(texture2D(colorTex, vec2(tc0.x,  tc12.y)).rgb, 1.0) * (w0.x  * w12.y) +
                   vec4(centerColor,                                      1.0) * (w12.x * w12.y) +
                   vec4(texture2D(colorTex, vec2(tc3.x,  tc12.y)).rgb, 1.0) * (w3.x  * w12.y) +
                   vec4(texture2D(colorTex, vec2(tc12.x, tc3.y )).rgb, 1.0) * (w12.x * w3.y );
	return color.rgb/color.a;

}

vec3 clip_aabb(vec3 q,vec3 aabb_min, vec3 aabb_max)
	{
		vec3 p_clip = 0.5 * (aabb_max + aabb_min);
		vec3 e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;

		vec3 v_clip = q - vec3(p_clip);
		vec3 v_unit = v_clip.xyz / e_clip;
		vec3 a_unit = abs(v_unit);
		float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

		if (ma_unit > 1.0)
			return vec3(p_clip) + v_clip / ma_unit;
		else
			return q;
	}												   

vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

vec3 tonemap(vec3 col){
	return pow(col/(1+luma(col)),vec3(1/2.232));
}
vec3 invTonemap(vec3 col){
	col = pow(col,vec3(2.232));
	return col/(1-luma(col));
}
vec3 RGB_YCoCg(vec3 c)
	{
		// Y = R/4 + G/2 + B/4
		// Co = R/2 - B/2
		// Cg = -R/4 + G/2 - B/4
		return vec3(
			 c.x/4.0 + c.y/2.0 + c.z/4.0,
			 c.x/2.0 - c.z/2.0,
			-c.x/4.0 + c.y/2.0 - c.z/4.0
		);
	}

	// https://software.intel.com/en-us/node/503873
	vec3 YCoCg_RGB(vec3 c)
	{
		// R = Y + Co - Cg
		// G = Y + Cg
		// B = Y - Co - Cg
		return max(vec3(
			c.x + c.y - c.z,
			c.x + c.z,
			c.x - c.y - c.z
		), 0.0);
	}					   

vec3 TAA_sspt(){
	//use velocity from the nearest texel from camera in a 3x3 box in order to improve edge quality in motion


	vec3 closestToCamera = vec3(texcoord,texture2D(depthtex2,texcoord).x);

	//reproject previous frame
	vec3 fragposition =	toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;
	previousPosition.xy = texcoord + velocity;


	//reject history if off-screen and early exit
	if (previousPosition.x < 0.0 || previousPosition.y < 0.0 || previousPosition.x > 1.0 || previousPosition.y > 1.0) return texture2D(colortex3, texcoord).rgb;

	//Samples current frame 3x3 neighboorhood
	vec3 albedoCurrent0 = texture2D(colortex3, texcoord).rgb;
  vec3 albedoCurrent1 = texture2D(colortex3, texcoord + vec2(texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent2 = texture2D(colortex3, texcoord + vec2(texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent3 = texture2D(colortex3, texcoord + vec2(-texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent4 = texture2D(colortex3, texcoord + vec2(-texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent5 = texture2D(colortex3, texcoord + vec2(0.0,texelSize.y)).rgb;
	vec3 albedoCurrent6 = texture2D(colortex3, texcoord + vec2(0.0,-texelSize.y)).rgb;
	vec3 albedoCurrent7 = texture2D(colortex3, texcoord + vec2(-texelSize.x,0.0)).rgb;
	vec3 albedoCurrent8 = texture2D(colortex3, texcoord + vec2(texelSize.x,0.0)).rgb;
	
	float vel = abs(velocity.x+velocity.y)*20;	
	
	

	//Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each channel in the 3x3 neighborhood
	vec3 cMax = max(max(max(albedoCurrent0,albedoCurrent1),albedoCurrent2),max(albedoCurrent3,max(albedoCurrent4,max(albedoCurrent5,max(albedoCurrent6,max(albedoCurrent7,albedoCurrent8))))));
	vec3 cMin = min(min(min(albedoCurrent0,albedoCurrent1),albedoCurrent2),min(albedoCurrent3,min(albedoCurrent4,min(albedoCurrent5,min(albedoCurrent6,min(albedoCurrent7,albedoCurrent8))))));


	vec3 albedoPrev = FastCatmulRom(colortex5, previousPosition.xy,vec4(texelSize, 1.0/texelSize), 0.82).xyz;
	vec3 finalcAcc = clamp(albedoPrev,cMin,cMax);



	//reduces blending factor if current texel is far from history, reduces flickering
	float lumDiff2 = distance(albedoPrev,finalcAcc)/luma(albedoPrev);
	lumDiff2 = 1.0-clamp(lumDiff2*lumDiff2,0.,1.)*SSPT_FLICKER_REDUCTION;

	//Increases blending factor when far from AABB and in motion, reduces ghosting
	float isclamped = distance(albedoPrev,finalcAcc)/luma(albedoPrev);
	float movementRejection = isclamped * clamp(length(velocity/texelSize),0.0,1.0)*0.5;
	//Blend current pixel with clamped history, apply fast tonemap beforehand to reduce flickering
	vec3 supersampled =   invTonemap(mix(tonemap(finalcAcc),tonemap(albedoCurrent0),clamp(SSPT_BLEND_FACTOR*lumDiff2 + movementRejection,0.,1.)));






	//De-tonemap
	return supersampled;
}

void main() {

/* DRAWBUFFERS:3 */

#ifdef SSPT 
	vec3 color = TAA_sspt();
	gl_FragData[0].rgb = clamp(fp10Dither(color,triangularize(interleaved_gradientNoise())),6.11*1e-5,65000.0);
#else
	vec3 color2 = texture2D(colortex3,texcoord).rgb;
	gl_FragData[0].rgb = color2;
#endif





}
