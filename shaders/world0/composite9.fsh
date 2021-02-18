#version 120
//Temporal Anti-Aliasing + Dynamic exposure calculations (vertex shader)

#extension GL_EXT_gpu_shader4 : enable

#include "/lib/res_params.glsl"


//TAA OPTIONS
//#define NO_CLIP	//Removes all anti-ghosting techniques used and creates a sharp image (good for still screenshots)
#define BLEND_FACTOR 0.01 //[0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.1 0.12 0.14 0.16] higher values = more flickering but sharper image, lower values = less flickering but the image will be blurrier
#define MOTION_REJECTION 0.0 //[0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5] //Higher values=sharper image in motion at the cost of flickering
#define ANTI_GHOSTING 0.0 //[0.0 0.25 0.5 0.75 1.0] High values reduce ghosting but may create flickering
#define FLICKER_REDUCTION 0.5  //[0.0 0.25 0.5 0.75 1.0] High values reduce flickering but may reduce sharpness
#define CLOSEST_VELOCITY //improves edge quality in motion at the cost of performance


const int noiseTextureResolution = 32;


/*
const int colortex0Format = RGBA16F;				// low res clouds (deferred->composite2) + low res VL (composite5->composite15)
const int colortex1Format = RGBA16;					//terrain gbuffer (gbuffer->composite2)
const int colortex2Format = RGBA16F;				//forward + transparencies (gbuffer->composite4)
const int colortex3Format = R11F_G11F_B10F;			//frame buffer + bloom (deferred6->final)
const int colortex4Format = RGBA16F;				//light values and skyboxes (everything)
const int colortex5Format = R11F_G11F_B10F;			//TAA buffer (everything)
const int colortex6Format = R11F_G11F_B10F;			//additionnal buffer for bloom (composite3->final)
const int colortex7Format = RGBA8;			//Final output, transparencies id (gbuffer->composite4)


const int colortex8Format = R11F_G11F_B10F;		
const int colortex9Format = R11F_G11F_B10F;	
		
const int colortexAFormat = RGBA16F;	
const int colortexBFormat = RGBA16F;	

const int colortexCFormat = RGBA16F;	
const int colortexDFormat = RGBA16F;	

const int colortexEFormat = RGBA16F;		//spec a
	
	


	
		
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
const bool colortexAClear = true;
const bool colortexBClear = false;


const bool colortexCClear = false;
const bool colortexDClear = true;
const bool colortexEClear = false;
const bool colortexFClear = false;



const bool shadowcolor0Clear = false;     
const bool shadowcolor1Clear = false;     



*/
varying vec2 texcoord;
flat varying float exposureA;
flat varying float tempOffsets;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex0;
uniform sampler2D colortex6;

uniform float near;
uniform float far;

uniform sampler2D colortexA;
uniform sampler2D colortexB;

uniform sampler2D colortexC;
uniform sampler2D colortexE;

uniform sampler2D colortexD;
uniform sampler2D depthtex0;

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

float ld(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}
float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}
float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+tempOffsets);
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
float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * frameCounter);
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
	return col/(1+luma(col));
}
vec3 invTonemap(vec3 col){
	return col/(1-luma(col));
}
const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
							vec2(-1.,3.)/8.,
							vec2(5.0,1.)/8.,
							vec2(-3,-5.)/8.,
							vec2(-5.,5.)/8.,
							vec2(-7.,-1.)/8.,
							vec2(3,7.)/8.,
							vec2(7.,-7.)/8.);
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
	vec3 closestToCamera = vec3(texcoord,texture2D(depthtex0,adjTC).x);
	#endif

	//reproject previous frame
	vec3 fragposition = toScreenSpace(closestToCamera);
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;
	previousPosition.xy = texcoord + velocity;

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
	vec3 albedoPrev = texture2D(colortex5, previousPosition.xy).xyz;
	vec3 supersampled =  mix(albedoPrev,albedoCurrent0,clamp(0.05,0.,1.));
	#endif

	//De-tonemap
	return supersampled;
}
#define DOF_MODE 0 // [0 1 2]
#define MANUAL_FOCUS	48.0	// If autofocus is turned off, sets the focus point (meters)	[0.06948345122280154 0.07243975703425146 0.07552184450877376 0.07873506526686186 0.0820849986238988 0.08557746127787037 0.08921851740926011 0.09301448921066349 0.09697196786440505 0.10109782498721881 0.10539922456186433 0.10988363537639657 0.11455884399268773 0.11943296826671962 0.12451447144412296 0.129812176855438 0.1353352832366127 0.1410933807013415 0.1470964673929768 0.15335496684492847 0.1598797460796939 0.16668213447794653 0.17377394345044514 0.18116748694692214 0.18887560283756183 0.19691167520419406 0.20528965757990927 0.21402409717744744 0.22313016014842982 0.2326236579172927 0.2425210746356487 0.25283959580474646 0.26359713811572677 0.27481238055948964 0.2865047968601901 0.29869468928867837 0.3114032239145977 0.32465246735834974 0.3384654251067422 0.3528660814588489 0.36787944117144233 0.3835315728763107 0.39984965434484737 0.4168620196785084 0.4345982085070782 0.453089017280169 0.4723665527410147 0.49246428767540973 0.513417119032592 0.5352614285189903 0.5580351457700471 0.5817778142098083 0.6065306597126334 0.6323366621862497 0.6592406302004438 0.6872892787909722 0.7165313105737893 0.7470175003104326 0.7788007830714049 0.8119363461506349 0.8464817248906141 0.8824969025845955 0.9200444146293233 0.9591894571091382 1.0 1.0425469051899914 1.086904049521229 1.1331484530668263 1.1813604128656459 1.2316236423470497 1.2840254166877414 1.338656724353094 1.3956124250860895 1.4549914146182013 1.5168967963882134 1.5814360605671443 1.6487212707001282 1.7188692582893286 1.7920018256557555 1.8682459574322223 1.9477340410546757 2.030604096634748 2.117000016612675 2.2070718156067044 2.300975890892825 2.398875293967098 2.5009400136621287 2.6073472713092674 2.718281828459045 2.833936307694169 2.9545115270921065 3.080216848918031 3.211270543153561 3.347900166492527 3.4903429574618414 3.638846248353525 3.7936678946831774 3.955076722920577 4.123352997269821 4.298788906309526 4.4816890703380645 4.672371070304759 4.871165999245474 5.0784190371800815 5.29449005047003 5.51975421667673 5.754602676005731 5.999443210467818 6.254700951936329 6.5208191203301125 6.798259793203881 7.087504708082256 7.38905609893065 7.703437568215379 8.031194996067258 8.372897488127265 8.72913836372013 9.10053618607165 9.487735836358526 9.891409633455755 10.312258501325767 10.751013186076355 11.208435524800691 11.685319768402522 12.182493960703473 12.700821376227164 13.241202019156521 13.804574186067095 14.391916095149892 15.00424758475255 15.642631884188171 16.30817745988666 17.00203994009402 17.725424121461643 18.479586061009854 19.265835257097933 20.085536923187668 20.940114358348602 21.831051418620845 22.75989509352673 23.728258192205157 24.737822143832553 25.790339917193062 26.88763906446752 28.03162489452614 29.22428378123494 30.46768661252054 31.763992386181833 33.11545195869231 34.52441195350251 35.99331883562839 37.524723159600995 39.12128399815321 40.78577355933337 42.52108200006278 44.3302224444953 46.21633621589248 48.182698291098816 50.23272298708815 52.36996988945491 54.598150033144236 56.92113234615337 59.34295036739207 61.867809250367884 64.50009306485578 67.24437240923179 70.10541234668786 73.08818067910767 76.19785657297057 79.43983955226133 82.81975887399955 86.3434833026695 90.01713130052181 93.84708165144015 97.83998453682129 102.00277308269969 106.34267539816554 110.86722712598126 115.58428452718766 120.50203812241894 125.62902691361414 130.9741532108186 136.54669808981876 142.35633750745257 148.4131591025766 154.72767971186107 161.3108636308289 168.17414165184545 175.32943091211476 182.78915558614753 190.56626845863 198.67427341514983 ]

#define focal  2.4
#define aperture  0.8	
flat varying vec2 rodExposureDepth;

void main() {

/* DRAWBUFFERS:5B */


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
	gl_FragData[1].r = (min(abs(aperture * (focal/100.0 * (z - focus)) / (z * (focus - focal/100.0))),texelSize.x*15.0));
	#ifdef TAA
	vec3 color = TAA_hq();



	gl_FragData[0].rgb = clamp(fp10Dither(color,triangularize(R2_dither())),6.11*1e-5,65000.0);



	
	#endif
	#ifndef TAA
	
	vec3 color = clamp(fp10Dither(texture2D(colortex3,texcoord).rgb,triangularize(interleaved_gradientNoise())),0.,65000.);

	gl_FragData[0].rgb = color;

	
	#endif





}
