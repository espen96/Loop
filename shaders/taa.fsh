#version 120
//Temporal Anti-Aliasing + Dynamic exposure calculations (vertex shader)

#extension GL_EXT_gpu_shader4 : enable

#include "/lib/res_params.glsl"
#include "/lib/kernel.glsl"



//TAA OPTIONS
//#define NO_CLIP	//Removes all anti-ghosting techniques used and creates a sharp image (good for still screenshots)
#define BLEND_FACTOR 0.01 //[0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.1 0.12 0.14 0.16] higher values = more flickering but sharper image, lower values = less flickering but the image will be blurrier
#define MOTION_REJECTION 0.0 //[0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5] //Higher values=sharper image in motion at the cost of flickering
#define ANTI_GHOSTING 0.0 //[0.0 0.25 0.5 0.75 1.0] High values reduce ghosting but may create flickering
#define FLICKER_REDUCTION 0.5  //[0.0 0.25 0.5 0.75 1.0] High values reduce flickering but may reduce sharpness
#define CLOSEST_VELOCITY //improves edge quality in motion at the cost of performance


const int noiseTextureResolution = 32;


/*
const int colortex0Format = RGBA16F;				
	// low res clouds (deferred->composite2) + low res VL (composite5->composite15)
const int colortex1Format = RGBA16;					
	//terrain gbuffer (gbuffer->composite2)
const int colortex2Format = RGBA16F;				
	//forward + transparencies (gbuffer->composite4)
const int colortex3Format = RGBA16F;			
	//frame buffer + bloom (deferred6->final)
const int colortex4Format = RGBA16F;				
	//light values and skyboxes (everything)
const int colortex5Format = RGB16F;			
	//TAA buffer (everything)
const int colortex6Format = R11F_G11F_B10F;			
	//additionnal buffer for bloom (composite3->final)
const int colortex7Format = RGBA8;					
	//Final output, transparencies id (gbuffer->composite4)
const int colortex8Format = RGBA16F;		   
	//transparent normals (gbuffer->composite0) ambient light(composite0->) and LAB emissive (a) (composite0->)
const int colortex9Format = RGBA16F;			
	//None
		
const int colortex10Format = RGBA16F;                
	//Normals (rgb),  LAB emissive (a) (gbuffer->composite0),  depth (a)  (composite0->)	
const int colortex11Format = R11F_G11F_B10F;	
	//Transparent normal for refraction(r) 
const int colortex12Format = RGBA32F;				
	//SSPT temporal
const int colortex13Format =  RG32UI;			    
	//motion vectors
const int colortex14Format = RGBA16F;			    
	//specular temporal
//const int colortex15Format = RGBA16F;		        
	//None
		
*/
//no need to clear the buffers, saves a few fps
/*
const bool colortex0Clear = false;
const bool colortex1Clear = false;
const bool colortex2Clear = true;
const bool colortex3Clear = false;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
const bool colortex7Clear = false;
const bool colortex8Clear = true;
const bool colortex9Clear = false;
const bool colortex10Clear = true;
const bool colortex11Clear = true;


const bool colortex12Clear = false;
const bool colortex13Clear = true;
const bool colortex14Clear = false;
const bool colortex15Clear = false;






*/
varying vec2 texcoord;
flat varying float exposureA;
flat varying float tempOffsets;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex0;
uniform sampler2D colortex1;
uniform sampler2D colortex6;
uniform sampler2D noisetex;
uniform sampler2D colortex15;
uniform sampler2D colortex2;


uniform float near;
uniform float far;

uniform sampler2D colortex10;
uniform sampler2D colortex11;

uniform sampler2D colortex12;
uniform sampler2D colortex14;

uniform sampler2D colortex13;
uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform float viewHeight;
uniform float viewWidth;
uniform int frameCounter;
uniform int framemod8;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferPreviousModelView;
#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
#include "/lib/projections.glsl"
#include "/lib/noise.glsl"
float ld(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}
float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
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


//returns the projected coordinates of the closest point to the camera in the 3x3 neighborhood
vec3 closestToCamera5taps(vec2 texcoord)
{
	vec2 du = vec2(texelSize.x*2., 0.0);
	vec2 dv = vec2(0.0, texelSize.y*2.);

	vec3 dtl = vec3(texcoord,0.) + vec3(-texelSize, texture2D(depthtex1, texcoord - dv - du).x);
	vec3 dtr = vec3(texcoord,0.) +  vec3( texelSize.x, -texelSize.y, texture2D(depthtex1, texcoord - dv + du).x);
	vec3 dmc = vec3(texcoord,0.) + vec3( 0.0, 0.0, texture2D(depthtex1, texcoord).x);
	vec3 dbl = vec3(texcoord,0.) + vec3(-texelSize.x, texelSize.y, texture2D(depthtex1, texcoord + dv - du).x);
	vec3 dbr = vec3(texcoord,0.) + vec3( texelSize.x, texelSize.y, texture2D(depthtex1, texcoord + dv + du).x);

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

//Modified texture interpolation from inigo quilez
vec4 smoothfilter(in sampler2D tex, in vec2 uv)
{
	vec2 textureResolution = vec2(viewWidth,viewHeight);
	uv = uv*textureResolution + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + fuv*fuv*fuv*(fuv*(fuv*6.0-15.0)+10.0);
	uv = (uv - 0.5)/textureResolution;
	return texture2D( tex, uv);
}
//Due to low sample count we "tonemap" the inputs to preserve colors and smoother edges
vec3 weightedSample(sampler2D colorTex, vec2 texcoord){
	vec3 wsample = texture2D(colorTex,texcoord).rgb*exposureA;
	return wsample/(1.0+luma(wsample));

}



//from : https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
vec4 SampleTextureCatmullRom(sampler2D tex, vec2 uv, vec2 texSize )
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * texSize;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 *= texelSize;
    texPos3 *= texelSize;
    texPos12 *= texelSize;

    vec4 result = vec4(0.0);
    result += texture2D(tex, vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += texture2D(tex, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += texture2D(tex, vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += texture2D(tex, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += texture2D(tex, vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += texture2D(tex, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
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

vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

vec3 tonemap(vec3 col){
	return col/(1+luma(col));
}
vec3 invTonemap(vec3 col){
	return col/(1-luma(col));
}
vec2 decodeVec2(float a){
    const vec2 constant1 = 65535. / vec2( 256., 65536.);
    const float constant2 = 256. / 255.;
    return fract( a * constant1 ) * constant2 ;
}
					
vec3 TAA_hq(){
	#ifdef TAA_UPSCALING
	vec2 adjTC = clamp(texcoord*RENDER_SCALE, vec2(0.0),RENDER_SCALE-texelSize*2.);
	#else
	vec2 adjTC = texcoord;
	#endif

	//use velocity from the nearest texel from camera in a 3x3 box in order to improve edge quality in motion
	#ifdef CLOSEST_VELOCITY
	vec3 closestToCamera = closestToCamera5taps(adjTC);
	#endif

	#ifndef CLOSEST_VELOCITY
	vec3 closestToCamera = vec3(texcoord,texture2D(depthtex1,adjTC).x);
	#endif

	vec4 data = texture2D(colortex1,adjTC);
	vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
	bool hand = abs(dataUnpacked1.w-0.75) <0.01;

	//reproject previous frame
	vec3 fragposition = toScreenSpace(closestToCamera);
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;
	previousPosition.xy = texcoord + velocity;
	if(hand) previousPosition.xy = texcoord;

	//reject history if off-screen and early exit
	if (previousPosition.x < 0.0 || previousPosition.y < 0.0 || previousPosition.x > 1.0 || previousPosition.y > 1.0)
		return smoothfilter(colortex3, adjTC + offsets[framemod8]*texelSize*0.5).xyz;

	#ifdef TAA_UPSCALING
	vec3 albedoCurrent0 = smoothfilter(colortex3, adjTC + offsets[framemod8]*texelSize*0.5).xyz;
	// Interpolating neighboorhood clampling boundaries between pixels
	vec3 cMax = texture2D(colortex0, adjTC).rgb;
	vec3 cMin = texture2D(colortex6, adjTC).rgb;
	#else
	vec3 albedoCurrent0 = texture2D(colortex3, adjTC).rgb;
	vec3 albedoCurrent1 = texture2D(colortex3, adjTC + vec2(texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent2 = texture2D(colortex3, adjTC + vec2(texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent3 = texture2D(colortex3, adjTC + vec2(-texelSize.x,-texelSize.y)).rgb;
	vec3 albedoCurrent4 = texture2D(colortex3, adjTC + vec2(-texelSize.x,texelSize.y)).rgb;
	vec3 albedoCurrent5 = texture2D(colortex3, adjTC + vec2(0.0,texelSize.y)).rgb;
	vec3 albedoCurrent6 = texture2D(colortex3, adjTC + vec2(0.0,-texelSize.y)).rgb;
	vec3 albedoCurrent7 = texture2D(colortex3, adjTC + vec2(-texelSize.x,0.0)).rgb;
	vec3 albedoCurrent8 = texture2D(colortex3, adjTC + vec2(texelSize.x,0.0)).rgb;
	//Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each channel in the 3x3 neighborhood
	vec3 cMax = max(max(max(albedoCurrent0,albedoCurrent1),albedoCurrent2),max(albedoCurrent3,max(albedoCurrent4,max(albedoCurrent5,max(albedoCurrent6,max(albedoCurrent7,albedoCurrent8))))));
	vec3 cMin = min(min(min(albedoCurrent0,albedoCurrent1),albedoCurrent2),min(albedoCurrent3,min(albedoCurrent4,min(albedoCurrent5,min(albedoCurrent6,min(albedoCurrent7,albedoCurrent8))))));
	albedoCurrent0 = smoothfilter(colortex3, adjTC + offsets[framemod8]*texelSize*0.5).rgb;
	#endif

	#ifndef NO_CLIP
	vec3 albedoPrev = max(FastCatmulRom(colortex5, previousPosition.xy,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
	vec3 finalcAcc = clamp(albedoPrev,cMin,cMax);

	//Increases blending factor when far from AABB and in motion, reduces ghosting
	float isclamped = distance(albedoPrev,finalcAcc)/luma(albedoPrev) * 0.5;
	float movementRejection = (0.12+isclamped)*clamp(length(velocity/texelSize),0.0,1.0);

	//Blend current pixel with clamped history, apply fast tonemap beforehand to reduce flickering
	vec3 supersampled = invTonemap(mix(tonemap(finalcAcc),tonemap(albedoCurrent0),clamp(BLEND_FACTOR + movementRejection,0.,1.)));
	#endif


	#ifdef NO_CLIP
	if (length(velocity) > 1e-6) return vec3(albedoCurrent0);
	vec3 albedoPrev = texture2D(colortex5, previousPosition.xy).xyz;
	vec3 supersampled =  mix(albedoPrev,albedoCurrent0,clamp(0.05,0.0,1.0));
	#endif

	//De-tonemap

		
	return supersampled;
}
	#define DOF_MODE 0 // [0 1 2]
	#define MANUAL_FOCUS 48.0 // If autofocus is turned off, sets the focus point (meters)	[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 5.0 6.0 7.0 8.0 10.0 12.0 14.0 16.0 20.0 24.0 28.0 32.0 40.0 48.0 56.0 64.0 92.0 128.0 192.0 256.0 384.0 512.0]

	#define focal 24 // Lens focal length in millimeters [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300]
#define aperture 16.0 // Aperture size in f-stops. Larger number = smaller aperture. [0.8 1.4 1.8 2.0 2.8 3.6 4.0 5.6 8.0 11.0 16.0 22.0]
	flat varying vec2 rodExposureDepth;

void main() {

/* RENDERTARGETS: 5 */


	#ifdef DOF
	float z = ld(texture2D(depthtex0, texcoord.st*RENDER_SCALE).r)*far;
		#if DOF_MODE == 0
			float focus = rodExposureDepth.y*far;
		#endif	
		#if DOF_MODE == 1
			float focus = MANUAL_FOCUS;
		#endif			
		
		#if DOF_MODE == 2
				focus = MANUAL_FOCUS * screenBrightness;
		#endif
		gl_FragData[1].r = (min(abs((focal / aperture) * (focal/1000.0 * (z - focus)) / (z * (focus - focal/1000.0))),texelSize.x*15.0));
	#endif

	#ifdef TAA
	vec3 color = TAA_hq();



	gl_FragData[0].rgb = color;



	
	#endif
 	#ifndef TAA
	
	vec3 color = clamp(fp10Dither(texture2D(colortex3,texcoord).rgb,triangularize(interleaved_gradientNoise())),0.,65000.);

	gl_FragData[0].rgb = color;

	
	#endif





}
