#version 140
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
//#define POM

#include "/lib/res_params.glsl"
#define SSAO

#define ROUGHREF
#define labemissivespwitch

//const bool colortex5MipmapEnabled = true;

#define Depth_Write_POM	// POM adjusts the actual position, so screen space shadows can cast shadows on POM
#define POM_DEPTH 0.25 // [0.025 0.05 0.075 0.1 0.125 0.15 0.20 0.25 0.30 0.50 0.75 1.0] //Increase to increase POM strength
#define CAVE_LIGHT_LEAK_FIX // Hackish way to remove sunlight incorrectly leaking into the caves. Can inacurrately create shadows in some places
//#define CLOUDS_SHADOWS
//#define LABSSS
#define CLOUDS_SHADOWS_STRENGTH 1.0 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define SPECSTRENGTH 1.0 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]

#define CLOUDS_QUALITY 0.35 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define TORCH_R 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_G 0.5 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_B 0.2 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Emissive_Strength 2.00 // [0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10 3.20 3.30 3.40 3.50 3.60 3.70 3.80 3.90 4.00 4.10 4.20 4.30 4.40 4.50 4.60 4.70 4.80 4.90 5.00 5.10 5.20 5.30 5.40 5.50 5.60 5.70 5.80 5.90 6.00 6.10 6.20 6.30 6.40 6.50 6.60 6.70 6.80 6.90 7.00 7.10 7.20 7.30 7.40 7.50 7.60 7.70 7.80 7.90 8.00 8.10 8.20 8.30 8.40 8.50 8.60 8.70 8.80 8.90 9.00 9.10 9.20 9.30 9.40 9.50 9.60 9.70 9.80 9.90 10.00 15.00 20.00 30.00 50.00 100.00 150.00 200.00]
#define SPEC_SSR_QUALITY 2 //[1 2 3 4 5 6 7 8 9 10 ]

const bool shadowHardwareFiltering = true;

flat varying vec4 lightCol; //main light source color (rgb),used light source(1=sun,-1=moon)
flat varying vec3 ambientUp;
flat varying vec3 ambientLeft;
flat varying vec3 ambientRight;
flat varying vec3 ambientB;
flat varying vec3 ambientF;
flat varying vec3 ambientDown;
flat varying vec3 WsunVec;
flat varying vec2 TAA_Offset;
flat varying float tempOffsets;
flat varying vec3 refractedSunVec;




uniform sampler2D colortex0;//clouds
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex2;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortex10;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;
uniform sampler2D colortex11;
uniform sampler2D colortex15;

uniform sampler2D colortex12;
uniform sampler2D colortex13;
uniform sampler2D colortex14;

uniform sampler2D colortex6; // Noise
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth


#ifdef SHADOWS_ON
uniform sampler2DShadow shadow;
#endif




uniform int framemod8;
uniform int heldBlockLightValue;
uniform int frameCounter;
uniform int isEyeInWater;
uniform float far;
uniform float wetness;
uniform float near;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 previousCameraPosition;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 gbufferModelView;
uniform float viewWidth;
uniform float viewHeight;
uniform float aspectRatio;
uniform vec2 texelSize;
uniform vec3 cameraPosition;
uniform vec3 sunVec;
uniform ivec2 eyeBrightnessSmooth;
#include "/lib/util.glsl"
#include "/lib/kernel.glsl"

vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
vec3 toScreenSpacePrev(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}


#include "/lib/noise.glsl"
#include "/lib/waterOptions.glsl"
#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/stars.glsl"
#include "/lib/volumetricClouds.glsl"



float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}


#include "/lib/specular.glsl"
vec3 normVec (vec3 vec){
	return vec*inversesqrt(dot(vec,vec));
}
float lengthVec (vec3 vec){
	return sqrt(dot(vec,vec));
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




vec3 decode (vec2 encn)
{
    vec3 unenc = vec3(0.0);
    encn = encn * 2.0 - 1.0;
    unenc.xy = abs(encn);
    unenc.z = 1.0 - unenc.x - unenc.y;
    unenc.xy = unenc.z <= 0.0 ? (1.0 - unenc.yx) * sign(encn) : encn;
    return normalize(unenc.xyz);
}

vec2 decodeVec2(float a){
    const vec2 constant1 = 65535. / vec2( 256., 65536.);
    const float constant2 = 256. / 255.;
    return fract( a * constant1 ) * constant2 ;
}
float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
	// l = (2*n)/(f+n-d(f-n))
	// f+n-d(f-n) = 2n/l
	// -d(f-n) = ((2n/l)-f-n)
	// d = -((2n/l)-f-n)/(f-n)

}
vec2 encode (vec3 unenc)
{
	unenc.xy = unenc.xy / dot(abs(unenc), vec3(1.0)) + 0.00390625;
	unenc.xy = unenc.z <= 0.0 ? (1.0 - abs(unenc.yx)) * sign(unenc.xy) : unenc.xy;
    vec2 encn = unenc.xy * 0.5 + 0.5;
	
    return vec2((encn));
}

float rayTraceShadow(vec3 dir,vec3 position,float dither){

    const float quality = 16.;
    vec3 clipPosition = toClipSpace3(position);
	//prevents the ray from going behind the camera
	float rayLength = ((position.z + dir.z * far*sqrt(3.)) > -near) ?
       (-near -position.z) / dir.z : far*sqrt(3.);
    vec3 direction = toClipSpace3(position+dir*rayLength)-clipPosition;  //convert to clip space
    direction.xyz = direction.xyz/max(abs(direction.x)/texelSize.x,abs(direction.y)/texelSize.y);	//fixed step size




    vec3 stepv = direction *10. * clamp(MC_RENDER_QUALITY,1.,2.0)*vec3(RENDER_SCALE,1.0);

	vec3 spos = clipPosition*vec3(RENDER_SCALE,1.0)+vec3(TAA_Offset*vec2(texelSize.x,texelSize.y)*0.5,0.0)+stepv;





	for (int i = 0; i < int(quality); i++) {
		spos += stepv*dither;

		float sp = texture
(depthtex1,spos.xy).x;
		if (sp >0.999) return 1.0;
        if( sp < spos.z) {

			float dist = abs(linZ(sp)-linZ(spos.z))/linZ(spos.z);

			if (dist < 0.035 ) return 0.0 + clamp(ld(sp)*10-1,0,1);



	}

	}
    return 1.0;
}



vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter,float distort)
{
		float alpha0 = sampleNumber/nb;
    float alpha = (sampleNumber+jitter)/nb;
    float angle = jitter*6.28 + alpha * 84.0 * 6.28;

    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*sqrt(alpha);
}



vec3 BilateralFiltering(sampler2D tex, sampler2D depth,vec2 coord,float frDepth,float maxZ){
  vec4 sampled = vec4(texelFetch2D(tex,ivec2(coord),0).rgb,1.0);

  return vec3(sampled.x,sampled.yz/sampled.w);
}

vec3 toShadowSpaceProjected(vec3 p3){
    p3 = mat3(gbufferModelViewInverse) * p3 + gbufferModelViewInverse[3].xyz;
    p3 = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
    p3 = diagonal3(shadowProjection) * p3 + shadowProjection[3].xyz;

    return p3;
}

float waterCaustics(vec3 wPos, vec3 lightSource){
	vec2 pos = (wPos.xz - lightSource.xz/lightSource.y*wPos.y)*4.0 ;
	vec2 movement = vec2(-0.02*frameTimeCounter);
	float caustic = 0.0;
	float weightSum = 0.0;
	float radiance =  2.39996;
	mat2 rotationMatrix  = mat2(vec2(cos(radiance),  -sin(radiance)),  vec2(sin(radiance),  cos(radiance)));
	vec2 displ = texture
(noisetex, pos*vec2(3.0,1.0)/96. + movement).bb*2.0-1.0;
	pos = pos*0.5+vec2(1.74*frameTimeCounter) ;
	for (int i = 0; i < 3; i++){
		pos = rotationMatrix * pos;
		caustic += Pow6(0.5+sin(dot(pos * exp2(0.8*i)+ displ*3.1415,vec2(0.5)))*0.5)*exp2(-0.8*i)/1.41;
		weightSum += exp2(-0.8*i);
	}
	return caustic * weightSum;
}

void waterVolumetrics(inout vec3 inColor, vec3 rayStart, vec3 rayEnd, float estEndDepth, float estSunDepth, float rayLength, float dither, vec3 waterCoefs, vec3 scatterCoef, vec3 ambient, vec3 lightSource, float VdotL){
		inColor *= exp(-rayLength * waterCoefs);	//No need to take the integrated value
		int spCount = rayMarchSampleCount;
		vec3 start = toShadowSpaceProjected(rayStart);
		vec3 end = toShadowSpaceProjected(rayEnd);
		vec3 dV = (end-start);
		//limit ray length at 32 blocks for performance and reducing integration error
		//you can't see above this anyway
		float maxZ = min(rayLength,32.0)/(1e-8+rayLength);
		dV *= maxZ;
		vec3 dVWorld = -mat3(gbufferModelViewInverse) * (rayEnd - rayStart) * maxZ;
		rayLength *= maxZ;
		estEndDepth *= maxZ;
		estSunDepth *= maxZ;
		vec3 absorbance = vec3(1.0);
		vec3 vL = vec3(0.0);
		float phase = phaseg(VdotL, Dirt_Mie_Phase);
		float expFactor = 11.0;
		vec3 progressW = gbufferModelViewInverse[3].xyz+cameraPosition;
		for (int i=0;i<spCount;i++) {
			float d = (pow(expFactor, float(i+dither)/float(spCount))/expFactor - 1.0/expFactor)/(1-1.0/expFactor);
			float dd = pow(expFactor, float(i+dither)/float(spCount)) * log(expFactor) / float(spCount)/(expFactor-1.0);
			vec3 spPos = start.xyz + dV*d;
			progressW = gbufferModelViewInverse[3].xyz+cameraPosition + d*dVWorld;
			//project into biased shadowmap space
			float distortFactor = calcDistort(spPos.xy);
			vec3 pos = vec3(spPos.xy*distortFactor, spPos.z);
			float sh = 1.0;
			if (abs(pos.x) < 1.0-0.5/2048. && abs(pos.y) < 1.0-0.5/2048){
				pos = pos*vec3(0.5,0.5,0.5*0.166)+0.5;


	#ifdef SHADOWS_ON
				sh =  shadow2D( shadow, pos).x;
		#else 
				sh =  0;
	#endif
				
			}
			vec3 ambientMul = exp(-estEndDepth * d * waterCoefs * 1.1);
			vec3 sunMul = exp(-estSunDepth * d * waterCoefs);
			vec3 light = (sh * lightSource*8./150./3.0 * phase * sunMul + ambientMul * ambient)*scatterCoef;
			vL += (light - light * exp(-waterCoefs * dd * rayLength)) / waterCoefs *absorbance;
			absorbance *= exp(-dd * rayLength * waterCoefs);
		}
		inColor += vL;
}
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}

vec2 tapLocation(int sampleNumber, float spinAngle,int nb, float nbRot,float r0)
{
    float alpha = (float(sampleNumber + r0) * (1.0 / (nb)));
    float angle = alpha * (nbRot * 6.28) + spinAngle*6.28;

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
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

vec3 closestToCamera5taps(vec2 texcoord)
{
	vec2 du = vec2(texelSize.x*2., 0.0);
	vec2 dv = vec2(0.0, texelSize.y*2.);

	vec3 dtl = vec3(texcoord,0.) + vec3(-texelSize, texture
(depthtex0, texcoord - dv - du).x);
	vec3 dtr = vec3(texcoord,0.) +  vec3( texelSize.x, -texelSize.y, texture
(depthtex0, texcoord - dv + du).x);
	vec3 dmc = vec3(texcoord,0.) + vec3( 0.0, 0.0, texture
(depthtex0, texcoord).x);
	vec3 dbl = vec3(texcoord,0.) + vec3(-texelSize.x, texelSize.y, texture
(depthtex0, texcoord + dv - du).x);
	vec3 dbr = vec3(texcoord,0.) + vec3( texelSize.x, texelSize.y, texture
(depthtex0, texcoord + dv + du).x);

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
vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
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
    vec3 centerColor = texture
(colorTex, vec2(tc12.x, tc12.y)).rgb;

    vec2 tc0 = rtMetrics.xy * (centerPosition - 1.0);
    vec2 tc3 = rtMetrics.xy * (centerPosition + 2.0);
    vec4 color = vec4(texture
(colorTex, vec2(tc12.x, tc0.y )).rgb, 1.0) * (w12.x * w0.y ) +
                   vec4(texture
(colorTex, vec2(tc0.x,  tc12.y)).rgb, 1.0) * (w0.x  * w12.y) +
                   vec4(centerColor,                                      1.0) * (w12.x * w12.y) +
                   vec4(texture
(colorTex, vec2(tc3.x,  tc12.y)).rgb, 1.0) * (w3.x  * w12.y) +
                   vec4(texture
(colorTex, vec2(tc12.x, tc3.y )).rgb, 1.0) * (w12.x * w3.y );
	return color.rgb/color.a;

}
vec3 tonemap(vec3 col){
	return col/(1+luma(col));
}
vec3 invTonemap(vec3 col){
	return col/(1-luma(col));
}		
//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 unenc, vec2 lightmaps)
{
	unenc.xy = unenc.xy / dot(abs(unenc), vec3(1.0)) + 0.00390625;
	unenc.xy = unenc.z <= 0.0 ? (1.0 - abs(unenc.yx)) * sign(unenc.xy) : unenc.xy;
    vec2 encn = unenc.xy * 0.5 + 0.5;
	
    return vec4((encn),vec2(lightmaps.x,lightmaps.y));
}
float remap_noise_tri_erp( const float v )
{
    float r2 = 0.5 * v;
    float f1 = sqrt( r2 );
    float f2 = 1.0 - sqrt( r2 - 0.25 );    
    return (v < 0.5) ? f1 : f2;
}


#define s2(a, b)				temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c)			s2(a, b); s2(a, c);
#define mx3(a, b, c)			s2(b, c); s2(a, c);

#define mnmx3(a, b, c)			mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)		s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)	s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges


#define vec vec3
#define toVec(x) x.rgb
vec3 median2(sampler2D tex1) {

    vec v[9];
    ivec2 ssC = ivec2(gl_FragCoord.xy);
	
	
    // Add the pixels which make up our window to the pixel array.
	
	
    for (int dX = -1; dX <= 1; ++dX) {
        for (int dY = -1; dY <= 1; ++dY) {
            ivec2 offset = ivec2(dX, dY);

            // If a pixel in the window is located at (x+dX, y+dY), put it at index (dX + R)(2R + 1) + (dY + R) of the
            // pixel array. This will fill the pixel array, with the top left pixel of the window at pixel[0] and the
            // bottom right pixel of the window at pixel[N-1].
			
			
            v[(dX + 1) * 3 + (dY + 1)] = toVec(texelFetch(tex1, ssC + offset, 0));
        }
    }

    vec temp;
    // Starting with a subset of size 6, remove the min and max each time
    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
    mnmx5(v[1], v[2], v[3], v[4], v[6]);
    mnmx4(v[2], v[3], v[4], v[7]);
    mnmx3(v[3], v[4], v[8]);
    vec3 result = v[4].rgb;
	
	return result;

}



vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3846153846) * direction;
  vec2 off2 = vec2(3.2307692308) * direction;
  color += texture
(image, uv) * 0.2270270270;
  color += texture
(image, uv + (off1 / resolution)) * 0.3162162162;
  color += texture
(image, uv - (off1 / resolution)) * 0.3162162162;
  color += texture
(image, uv + (off2 / resolution)) * 0.0702702703;
  color += texture
(image, uv - (off2 / resolution)) * 0.0702702703;
  return color;
}
void PromoOutline(inout vec3 color, sampler2D depth) {
		vec2 texCoord = gl_FragCoord.xy*texelSize;
	float ph = 1.0 / viewHeight;
	float pw = ph / aspectRatio;

	float outlinea = 0.0, outlineb = 0.0, outlinec = 0.0, outlined = 0.0;
	float z = ld(texture
(depth, texCoord).r) * far;
	float totalz = 0.0;
	float maxz = z;
	float sampleza = 0.0;
	float samplezb = 0.0;

	for(int i = 0; i < 12; i++) {
		vec2 offset = vec2(pw, ph) * outlineOffsets[i];
		sampleza = ld(texture
(depth, texCoord + offset).r) * far;
		samplezb = ld(texture
(depth, texCoord - offset).r) * far;
		maxz = max(maxz, max(sampleza, samplezb));

		float sample = (z * 2.0 - (sampleza + samplezb)) / length(outlineOffsets[i]);

		outlinea += clamp(1.0 + sample * 4.0 / z, 0.0, 1.0);
		if(i >= 8) outlineb += 1.0 - (1.0 - clamp(1.0 - sample * 512.0 / z, 0.0, 1.0)) * clamp(1.0 - sample * 16.0 / z, 0.0, 1.0);
		outlinec += clamp(1.0 + sample * 128.0 / z, 0.0, 1.0);

		totalz += sampleza + samplezb;
	}
	outlinea = clamp(outlinea - 10.0, 0.0, 1.0);
	outlineb = clamp(outlineb - 2.0, 0.0, 1.0);
	outlinec = 1.0 - clamp(outlinec - 11.0, 0.0, 1.0);
	outlined = clamp(1.0 + 64.0 * (z - maxz) / z, 0.0, 1.0);
	
	float outline = clamp((0.2 * outlinea * outlineb + 0.8) + 0.2 * outlinec * outlined,1,1.01);

	color = sqrt(sqrt(color));
	color *= outline;
	color *= color; 
	color *= color;
}

vec3 getProjPos(in vec2 uv, in float depth) {
    return vec3(uv, depth) * 2.0 - 1.0;
}

vec3 proj2view(in vec3 proj_pos) {
    vec4 view_pos = gbufferProjectionInverse * vec4(proj_pos, 1.0);
    return view_pos.xyz / view_pos.w;
}


#define viewMAD(m, v) (mat3(m) * (v) + (m)[3].xyz)

vec3 reproject(vec3 sceneSpace, bool hand) {
    vec3 prevScreenPos = hand ? vec3(0.0) : cameraPosition - previousCameraPosition;
    prevScreenPos = sceneSpace + prevScreenPos;
    prevScreenPos = viewMAD(gbufferPreviousModelView, prevScreenPos);
    prevScreenPos = viewMAD(gbufferPreviousProjection, prevScreenPos) * (0.5 / -prevScreenPos.z) + 0.5;

    return prevScreenPos;
}

	float checkerboard(in vec2 uv)
{
    vec2 pos = floor(uv);
  	return mod(pos.x + mod(pos.y, 2.0), 2.0);
}	


void main() {






	vec2 texcoord = gl_FragCoord.xy*texelSize;
	float dirtAmount = Dirt_Amount;
	vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
	vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
	vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
	vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B);
	float z0 = texture
(depthtex0,texcoord).x;
	float z = texture
(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();

	vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normalize(p3);

	
	
	
	
	//sky
	if (z >=1.0) {
		vec3 color = vec3(1.0,0.4,0.2)/4000.0*1050.0*0.1;
		vec4 cloud = texture
_bicubic(colortex0,texcoord*CLOUDS_QUALITY);
		color = (color*cloud.a+cloud.rgb)*0.0;
		gl_FragData[0].rgb = clamp(fp10Dither(color*8./3. * (1.0-rainStrength*0.4),triangularize(noise)),0.0,65000.);
		//if (gl_FragData[0].r > 65000.) 	gl_FragData[0].rgb = vec3(0.0);
		vec4 trpData = texture
(colortex7,texcoord);
		bool iswater = texture
(colortex7,texcoord).a > 0.99;
		if (iswater){
			vec3 fragpos0 = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z0));
			float Vdiff = distance(fragpos,fragpos0);
			float VdotU = np3.y;
			float estimatedDepth = Vdiff * abs(VdotU);	//assuming water plane
			float estimatedSunDepth = estimatedDepth/abs(WsunVec.y); //assuming water plane

			vec3 lightColVol = lightCol.rgb * (0.91-pow(1.0-WsunVec.y,5.0)*0.86);	//fresnel
			vec3 ambientColVol = ambientUp*8./150./3.*0.84*2.0/pi * eyeBrightnessSmooth.y / 240.0;
			if (isEyeInWater == 0)
				waterVolumetrics(gl_FragData[0].rgb, fragpos0, fragpos, estimatedDepth, estimatedSunDepth, Vdiff, noise, totEpsilon, scatterCoef, ambientColVol, lightColVol, dot(np3, WsunVec));
		}
	}
	//land
	else {
		p3 += gbufferModelViewInverse[3].xyz;

		vec4 trpData = texture
(colortex7,texcoord);
		vec3 preshade = texture
(colortex3,texcoord).rgb;
		vec4 transparent = texture
(colortex2,texcoord);
		
		bool iswater = texture
(colortex7,texcoord).a > 0.99;
		bool istransparent = luma(transparent.rgb) > 0.1;

//		float edgemask = clamp(edgefilter(texcoord*RENDER_SCALE,2,colortex8).rgb,0,1).r;
		vec4 data = texture
(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		
		
		vec4 shadowCol = vec4(0.0);
		vec4 shadowCol2 = vec4(0.0);
		float caustics = 1;
		
		
		vec3 normal = mat3(gbufferModelViewInverse) * worldToView(decode(dataUnpacked0.yw));
//	    	 normal = mat3(gbufferModelViewInverse) * blur5(colortex10, texcoord, vec2(viewWidth,viewHeight), vec2(1,1) ).rgb;


	//	vec3 normal = mat3(gbufferModelViewInverse) * texture
(colortex10,texcoord).rgb;
		vec3 normal2 =  worldToView(decode(dataUnpacked0.yw));


		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		vec2 lightmap = dataUnpacked1.yz;
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;	// Strong translucency
		bool translucent2 = abs(dataUnpacked1.w-0.6) <0.01;	// Weak translucency
		float labsss = trpData.z;


			if (labsss < 64.5/255.0)
				labsss = 0.0;

		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		float NdotLGeom = dot(normal, WsunVec);
		float NdotL = NdotLGeom;
		if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater == 1))
			NdotL = dot(normal, refractedSunVec);

		float diffuseSun = clamp(NdotL,0.,1.0);
		vec2 filtered = vec2(1.0,0.0);
		vec3 caustic = vec3(1.412,1.0,0.0);
		if (!hand){
			filtered = decodeVec2(texture
(colortex3,texcoord).a);
		}
		float shading = 1.0 - filtered.y;


		vec3 SSS = vec3(0.0);
		float sssAmount = 0.0;
		#ifdef Variable_Penumbra_Shadows
		// compute shadows only if not backfacing the sun
		// or if the blocker search was full or empty
		// always compute all shadows at close range where artifacts may be more visible
		if (diffuseSun > 0.001 &&!hand) {
		#else
		if (translucent) {
			sssAmount = 0.5;
			diffuseSun = mix(max(phaseg(dot(np3, WsunVec),0.5), 2.0*phaseg(dot(np3, WsunVec),0.1))*3.14150*1.6, diffuseSun, 0.3);
		}
		if (diffuseSun > 0.000) {
		#endif

		}
		//custom shading model for translucent objects
		#ifdef Variable_Penumbra_Shadows
		#ifndef LABSSS
		if (translucent) {
			sssAmount = 0.5;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.x*11.0*extinction) + 3.0*exp(-filtered.x*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.5*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			SSS *= sqrt(lightmap.y);
		}

		if (translucent2) {
			sssAmount = 0.2;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.x*11.0*extinction) + 3.0*exp(-filtered.x*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			SSS *= sqrt(lightmap.y);
		}
		
		#else
			if (labsss > 0.0) {
			sssAmount = labsss;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.x*11.0*extinction) + 3.0*exp(-filtered.x*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			SSS *= clamp(sqrt(lightmap.x*2-1.5),0,1);
		}	
		#endif
		#endif


		float labemissive = texture
(colortex8, texcoord).a;
		vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));
		vec3 ambientLight = ambientUp*mix(clamp(ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientDown*mix(clamp(-ambientCoefs.y,0.,1.), 0.166, sssAmount);
		ambientLight += ambientRight*mix(clamp(ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientLeft*mix(clamp(-ambientCoefs.x,0.,1.), 0.166, sssAmount);
		ambientLight += ambientB*mix(clamp(ambientCoefs.z,0.,1.), 0.166, sssAmount);
		ambientLight += ambientF*mix(clamp(-ambientCoefs.z,0.,1.), 0.166, sssAmount);

		vec3 directLightCol = lightCol.rgb*2;
		vec3 custom_lightmap = texture
(colortex4,(lightmap*15.0+0.5+vec2(0.0,19.))*texelSize).rgb*10./150./3.;
		float emitting = 0.0;
		if (emissive || (hand && heldBlockLightValue > 0.1)){



		if(!hand)	emitting = (luma(albedo)*1.0*Emissive_Strength);
		if (hand)   emitting = (luma(albedo)*Emissive_Strength)*2-1;
			custom_lightmap.y = 0.0;
			emitting = clamp(emitting*(1-luma(transparent.rgb*20)),0.0,10);






		}

	#ifdef SPEC
		emitting =  (labemissive*Emissive_Strength);
	#endif				
	            if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater == 1)){
			vec3 fragpos0 = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z0));
			float Vdiff = distance(fragpos,fragpos0);
			float VdotU = np3.y;
			float estimatedDepth = Vdiff * abs(VdotU);	//assuming water plane
			if (isEyeInWater == 1){
				Vdiff = length(fragpos);
				estimatedDepth =  clamp((15.5-lightmap.y*16.0)/15.5,0.,1.0);
				estimatedDepth *= estimatedDepth*estimatedDepth*32.0;
				#ifndef lightMapDepthEstimation
					estimatedDepth = max(Water_Top_Layer - (cameraPosition.y+p3.y),0.0);
				#endif
			}
			float estimatedSunDepth = estimatedDepth/abs(refractedSunVec.y); //assuming water plane
			directLightCol *= exp(-totEpsilon*estimatedSunDepth)*(1.0-Pow5(1.0-WsunVec.y));
			float caustics = waterCaustics(mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz + cameraPosition, refractedSunVec);
			directLightCol *= mix(caustics*0.5+0.5,1.0,exp(-estimatedSunDepth/3.0));

			if (isEyeInWater == 0){
				ambientLight *= min(exp(-totEpsilon*estimatedDepth), custom_lightmap.x);
				ambientLight += custom_lightmap.z;
			}
			else {
				ambientLight += 10.0 * exp(-totEpsilon*8.0);
				ambientLight *= exp(-totEpsilon*estimatedDepth)*8./150./3.;
			}
			ambientLight *= mix(caustics,1.0,0.85);
			ambientLight += custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B);

			//combine all light sources
			gl_FragData[1].rgb = vec3(0.0);
			gl_FragData[0].rgb = ((preshade)/pi*8./150./3.*directLightCol.rgb + ambientLight + emitting) * albedo;
			//Bruteforce integration is probably overkill
			vec3 lightColVol = lightCol.rgb * (1.0-Pow5(1.0-WsunVec.y));	//fresnel
			vec3 ambientColVol =  ambientUp*8./150./3.*0.5 / 240.0 * eyeBrightnessSmooth.y;
			if (isEyeInWater == 0)
				waterVolumetrics(gl_FragData[0].rgb, fragpos0, fragpos, estimatedDepth, estimatedSunDepth, Vdiff, noise, totEpsilon, scatterCoef, ambientColVol, lightColVol, dot(np3, WsunVec));

		}
		else {


				



			ambientLight = texture
(colortex8,texcoord).rgb;
		#ifdef ssptfilter
			ambientLight = mix( median2(colortex8), ambientLight,0.0);
		#endif	
			ambientLight *= (1+clamp(transparent.rgb*10*emitting*2,1,100));					
					



//	ambientLight*= blur5(colortex15, texcoord, vec2(viewWidth,viewHeight), vec2(1,0) ).r;

			//combine all light sources
	#ifdef SSGI
//	float labao =	texture
(colortex11, gl_FragCoord.xy*texelSize).g;
	float labao =	1;
	#else 
	float labao = 1;
	#endif

			gl_FragData[0].rgb = (((preshade)/pi*8./150./3.*directLightCol.rgb + (ambientLight*labao) + emitting)*albedo) ;


	#define albedotint
			
			// Speculars
	#ifdef SPEC

			// Speculars
			// Unpack labpbr
			float roughness = unpackRoughness(trpData.x);
			float porosity = trpData.z;
			float rej = 1;
		//	float edgemask = clamp(edgefilter(texcoord*RENDER_SCALE,2,colortex8).rgb,0,1).r;

			if (porosity > 64.5/255.0)
				porosity = 0.0;
			porosity = porosity*255.0/64.0;
			vec3 f0 = vec3(trpData.y);

	#ifdef albedotint
			if (f0.y > 229.5/255.0){
				f0 = albedo;
			}
	#else
			if (f0.y > 229.5/255.0){
			f0.rgb *= MetalCol(f0.y);	
	//			else f0 *= albedo.rgb;			


			}	
	#endif
			
			float rainMult = sqrt(lightmap.y)*wetness*(1.0-square(porosity));
			roughness = mix(roughness, 0.01, rainMult);

			f0 = mix(f0, vec3(0.02), rainMult);
			//f0 = vec3(0.5);
			//roughness = 0.01;

			// Energy conservation between diffuse and specular
			vec3 fresnelDiffuse = vec3(0.0);
			vec3 reflection = vec3(0.0);

			// Sun specular
			vec3 specTerm = shading * GGX2(normal, -np3,  WsunVec, roughness+0.05*0.95, f0) * 8./150./3.;

			vec3 indirectSpecular = vec3(0.0);
			const int nSpecularSamples = SPEC_SSR_QUALITY;
			mat3 basis = CoordBase(normal);
			vec3 normSpaceView = -np3*basis;
			vec3 rayContrib = vec3(0.0);
			vec3 reflectedVector = reflect(normalize(fragpos), normalize(normal2));
		
			for (int i = 0; i < nSpecularSamples; i++){
				// Generate ray
				int seed = frameCounter*nSpecularSamples + i;
				vec2 ij = fract(R2_samples(seed) + blueNoise());
				vec3 H = sampleGGXVNDF(normSpaceView, roughness, roughness, ij.x, ij.y);
				vec3 Ln = reflect(-normSpaceView, H);
				vec3 L = basis * Ln;
		//		if (dot(reflectedVector, normal2) < 0.0) L = -L;	
				// Ray contribution
				float g1 = g(clamp(dot(normal, L),0.0,1.0), roughness);
				vec3 F = f0 + (1.0 - f0) * pow(clamp(1.0 + dot(-Ln, H),0.0,1.0), 5.0);
 // 				gl_FragData[0].rgb = vec3((texture
Lod(colortex5,texcoord,4).rgb));

				     rayContrib = F * g1;

				// Skip calculations if ray does not contribute much to the lighting
		
				if (luma(rayContrib) > 0.05){
				
					vec4 reflection = vec4(0.0,0.0,0.0,0.0);
					// Scale quality with ray contribution
					float rayQuality = 35*sqrt(luma(rayContrib));

					// Skip SSR if ray contribution is low
					if (rayQuality > 5.0) {



						vec3 rtPos = rayTrace(normal2,mat3(gbufferModelView) * L, fragpos.xyz, noise, rayQuality);
				//		vec3 rtPos = rayTrace(normal2,reflectedVector, fragpos.xyz, noise, rayQuality);
		
						// Reproject on previous frame
						if (rtPos.z < 1.){
							vec3 previousPosition = mat3(gbufferModelViewInverse) * toScreenSpace(rtPos) + gbufferModelViewInverse[3].xyz + cameraPosition-previousCameraPosition;
							previousPosition = mat3(gbufferPreviousModelView) * previousPosition + gbufferPreviousModelView[3].xyz;
							previousPosition.xy = projMAD(gbufferPreviousProjection, previousPosition).xy / -previousPosition.z * 0.5 + 0.5;
							if (previousPosition.x > 0.0 && previousPosition.y > 0.0 && previousPosition.x < 1.0 && previousPosition.x < 1.0) {
								reflection.a = 1.0;
								reflection.rgb = clamp(clamp(texture
(colortex5,previousPosition.xy).rgb,0.0, luma(texture
(colortex5,previousPosition.xy).rgb)),0,20);
							}
						}
					}

					// Sample skybox
					if (reflection.a < 0.9){
						reflection.rgb = clamp((skyCloudsFromTex(L, colortex4).rgb)*clamp(lightmap.y-0.8,0,1),0,10);
						reflection.rgb *= sqrt(lightmap.y)/150.*8./3.;
					}
					indirectSpecular += reflection.rgb * rayContrib;
					fresnelDiffuse += rayContrib;
				}

			}
			
			vec4    historyGData    = vec4(1.0);
			vec4   indirectHistory = vec4(0.0);
			vec4    indirectCurrent = texture
(colortex5,texcoord.xy/RENDER_SCALE).rgba;
			float    sceneDepth = texture
(depthtex0,texcoord.xy).x;
			vec3 scenePos   = viewMAD(gbufferModelViewInverse,fragpos);
			vec3 reprojection   = reproject(scenePos, hand);	
	        bool offscreen      = clamp(reprojection,0,1) != reprojection;
			vec3 closestToCamera = closestToCamera5taps(texcoord);
			vec3 fragposition = toScreenSpace(closestToCamera);			
			fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
			vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
			previousPosition = toClipSpace3Prev(previousPosition);
			vec2 velocity = previousPosition.xy - closestToCamera.xy;
			previousPosition.xy = texcoord + velocity;

			vec3 albedoCurrent0 = texture
(colortex14, reprojection.xy*RENDER_SCALE).rgb;
			vec3 albedoCurrent1 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(texelSize.x,texelSize.y)).rgb;
			vec3 albedoCurrent2 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(texelSize.x,-texelSize.y)).rgb;
			vec3 albedoCurrent3 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(-texelSize.x,-texelSize.y)).rgb;
			vec3 albedoCurrent4 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(-texelSize.x,texelSize.y)).rgb;
			vec3 albedoCurrent5 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(0.0,texelSize.y)).rgb;
			vec3 albedoCurrent6 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(0.0,-texelSize.y)).rgb;
			vec3 albedoCurrent7 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(-texelSize.x,0.0)).rgb;
			vec3 albedoCurrent8 = texture
(colortex14, reprojection.xy*RENDER_SCALE + vec2(texelSize.x,0.0)).rgb;

			//Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each channel in the 3x3 neighborhood

			vec3 cMax = max(max(max(albedoCurrent0,albedoCurrent1),albedoCurrent2),max(albedoCurrent3,max(albedoCurrent4,max(albedoCurrent5,max(albedoCurrent6,max(albedoCurrent7,albedoCurrent8))))));
			vec3 cMin = min(min(min(albedoCurrent0,albedoCurrent1),albedoCurrent2),min(albedoCurrent3,min(albedoCurrent4,min(albedoCurrent5,min(albedoCurrent6,min(albedoCurrent7,albedoCurrent8))))));

			
					 
			vec3 albedoPrev = max(FastCatmulRom(colortex14, reprojection.xy*RENDER_SCALE,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
			vec3 albedoPrev2 = max(FastCatmulRom(colortex5, reprojection.xy*RENDER_SCALE/RENDER_SCALE,vec4(texelSize, 1.0/texelSize), 0.75).xyz, 0.0);
			vec3 finalcAcc = clamp(albedoPrev,cMin,cMax);			
					
			float isclamped = (clamp(clamp(((distance(albedoPrev,finalcAcc)/luma(albedoPrev))),0,10),0,10));	 
			float isclamped2 = (((distance(albedoPrev2,finalcAcc)/luma(albedoPrev2)) *0.9) );	 
			float isclamped3 = (((distance(luma(albedoPrev2),texture
(colortex14,reprojection.xy*RENDER_SCALE).a)/luma(albedoPrev2)) *0.5) );	 
			float clamped = dot(isclamped,isclamped2);
			rej = clamp( (clamp(   (isclamped3)   ,0,1) ) +((isclamped)*clamp(length(velocity/texelSize),0.0,1.0))    ,0.5,1);	
			vec3 bn = blueNoise(gl_FragCoord.xy).xyz;
			if(hand) rej = 1;
		
			
		//	vec3 speculars = mix( (((indirectSpecular) /nSpecularSamples + specTerm * directLightCol.rgb)),vec3(0.0), clamp(rej,0,1.0) );			
		reflection = mix(texture
(colortex14, reprojection.xy*RENDER_SCALE).rgb,(((indirectSpecular) /nSpecularSamples + specTerm * directLightCol.rgb)), rej );			
	
		gl_FragData[1].rgb = reflection;
	//	gl_FragData[0].rgb = (indirectSpecular/nSpecularSamples + specTerm * directLightCol.rgb)*SPECSTRENGTH +  (1.0-fresnelDiffuse/nSpecularSamples*0.6) * gl_FragData[0].rgb;

		if (!hand)	gl_FragData[0].rgb =   (reflection) +  (1.0-fresnelDiffuse/(nSpecularSamples*1.22)) * gl_FragData[0].rgb;
		

		#endif
//		if (!hand)	gl_FragData[2].rgb =	vec3( gl_FragData[1].rgb );
	
	
		}



//vec3 color22 = 	gl_FragData[0].rgb;
//PromoOutline(  color22,  depthtex0);
//gl_FragData[0].rgb =  mix(color22,gl_FragData[0].rgb,ld(z));


	
	}	


/* RENDERTARGETS: 3,14 */
}
