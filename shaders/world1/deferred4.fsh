#version 150 compatibility
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
//#define POM

#include "/lib/res_params.glsl"
#define SSAO

#define ROUGHREF
#define labemissivespwitch


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

flat in vec4 lightCol; //main light source color (rgb),used light source(1=sun,-1=moon)
flat in vec3 WsunVec;
flat in vec2 TAA_Offset;
flat in float tempOffsets;
flat in vec3 refractedSunVec;




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
uniform sampler2D colortex14;

uniform sampler2D colortex6; // Noise
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth
uniform sampler2D colortex13;


#ifdef SHADOWS_ON
uniform sampler2DShadow shadow;

uniform sampler2DShadow shadowtex1;
uniform sampler2DShadow shadowtex0;
uniform sampler2DShadow shadowcolor0;

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




    vec3 stepv = direction *3. * 1.0 *vec3(RENDER_SCALE,1.0);

	vec3 spos = clipPosition*vec3(RENDER_SCALE,1.0)+vec3(TAA_Offset*vec2(texelSize.x,texelSize.y)*0.5,0.0)+stepv*dither;





	for (int i = 0; i < int(quality); i++) {
		spos += stepv;

		float sp = texture(depthtex1,spos.xy).x;
        if( sp < spos.z) {

			float dist = abs(linZ(sp)-linZ(spos.z))/linZ(spos.z);

			if (dist < 0.01 ) return 0.0;



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
	vec2 displ = texture(noisetex, pos*vec2(3.0,1.0)/96. + movement).bb*2.0-1.0;
	pos = pos*0.5+vec2(1.74*frameTimeCounter) ;
	for (int i = 0; i < 3; i++){
		pos = rotationMatrix * pos;
		caustic += Pow6(0.5+sin(dot(pos * exp2(0.8*i)+ displ*3.1415,vec2(0.5)))*0.5)*exp2(-0.8*i)/1.41;
		weightSum += exp2(-0.8*i);
	}
	return caustic * weightSum;
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

	vec3 dtl = vec3(texcoord,0.) + vec3(-texelSize, texture(depthtex0, texcoord - dv - du).x);
	vec3 dtr = vec3(texcoord,0.) +  vec3( texelSize.x, -texelSize.y, texture(depthtex0, texcoord - dv + du).x);
	vec3 dmc = vec3(texcoord,0.) + vec3( 0.0, 0.0, texture(depthtex0, texcoord).x);
	vec3 dbl = vec3(texcoord,0.) + vec3(-texelSize.x, texelSize.y, texture(depthtex0, texcoord + dv - du).x);
	vec3 dbr = vec3(texcoord,0.) + vec3( texelSize.x, texelSize.y, texture(depthtex0, texcoord + dv + du).x);

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
    vec3 centerColor = texture(colorTex, vec2(tc12.x, tc12.y)).rgb;

    vec2 tc0 = rtMetrics.xy * (centerPosition - 1.0);
    vec2 tc3 = rtMetrics.xy * (centerPosition + 2.0);
    vec4 color = vec4(texture(colorTex, vec2(tc12.x, tc0.y )).rgb, 1.0) * (w12.x * w0.y ) +
                   vec4(texture(colorTex, vec2(tc0.x,  tc12.y)).rgb, 1.0) * (w0.x  * w12.y) +
                   vec4(centerColor,                                      1.0) * (w12.x * w12.y) +
                   vec4(texture(colorTex, vec2(tc3.x,  tc12.y)).rgb, 1.0) * (w3.x  * w12.y) +
                   vec4(texture(colorTex, vec2(tc12.x, tc3.y )).rgb, 1.0) * (w12.x * w3.y );
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
  color += texture(image, uv) * 0.2270270270;
  color += texture(image, uv + (off1 / resolution)) * 0.3162162162;
  color += texture(image, uv - (off1 / resolution)) * 0.3162162162;
  color += texture(image, uv + (off2 / resolution)) * 0.0702702703;
  color += texture(image, uv - (off2 / resolution)) * 0.0702702703;
  return color;
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
vec2 moment(ivec2 pos) {

float weightSum = 1.0;
vec2 moment = texelFetch(colortex15, pos, 0).rg;
float depth = texelFetch(depthtex0, pos, 0).x;
vec3 normal = texelFetch(colortex10, pos, 0).xyz;
    for (int i = 0; i<9; i++) {
        ivec2 deltaPos     = kernelO_3x3[i]*4;
        //  We already have the center data
//        if (pos != 0 && pos != 0) { continue; }

        //  Sample current point data with current uv
        ivec2 p = pos + deltaPos;
        vec4 curColor = texelFetch(colortex12, ivec2(p), 0);       
		float curDepth = texelFetch(depthtex0, p, 0).x;
        vec3 curNormal = texelFetch(colortex10, p, 0).xyz;

        //  Determine the average brightness of this sample
        //  Using International Telecommunications Union's ITU BT.601 encoding params
        float l = luma(curColor.rgb);

        float weightDepth = abs(curDepth - depth) / (depth * length(vec2(deltaPos)) + 1.0e-2);
        float weightNormal = pow(max(0, dot(curNormal, normal)), 32.0);
        float w = exp(-weightDepth) * weightNormal;


        weightSum += w;

        moment += vec2(l, l * l) * w;
    }


moment /= weightSum;

    return  moment;
}

float encodeVec2v2(vec2 a){
    ivec2 bf = ivec2(a*255.);
    return float( bf.x|(bf.y<<8) ) / 65535.;
}









void main() {
	vec2 texcoord = gl_FragCoord.xy*texelSize;
	
	float dirtAmount = Dirt_Amount;
	vec3 waterEpsilon = vec3(Water_Absorb_R, Water_Absorb_G, Water_Absorb_B);
	vec3 dirtEpsilon = vec3(Dirt_Absorb_R, Dirt_Absorb_G, Dirt_Absorb_B);
	vec3 totEpsilon = dirtEpsilon*dirtAmount + waterEpsilon;
	vec3 scatterCoef = dirtAmount * vec3(Dirt_Scatter_R, Dirt_Scatter_G, Dirt_Scatter_B);
	float z0 = texture(depthtex0,texcoord).x;
	float z = texture(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();

	vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normalize(p3);

	
	
	
	
	//sky
	if (z < 1.0) {

		p3 += gbufferModelViewInverse[3].xyz;

		vec4 trpData = texture(colortex7,texcoord);
		vec4 transparent = texture(colortex2,texcoord);
		
		bool iswater = texture(colortex7,texcoord).a > 0.99;
		bool istransparent = luma(transparent.rgb) > 0.1;

//		float edgemask = clamp(edgefilter(texcoord*RENDER_SCALE,2,colortex8).rgb,0,1).r;
		vec4 data = texture(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		
		
		vec4 shadowCol = vec4(0.0);
		vec4 shadowCol2 = vec4(0.0);
		float caustics = 1;
		
		
		vec3 normal = mat3(gbufferModelViewInverse) * worldToView(decode(dataUnpacked0.yw));
//	    	 normal = mat3(gbufferModelViewInverse) * blur5(colortex10, texcoord, vec2(viewWidth,viewHeight), vec2(1,1) ).rgb;


	//	vec3 normal = mat3(gbufferModelViewInverse) * texture(colortex10,texcoord).rgb;
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
		vec3 filtered = vec3(1.412,1.0,0.0);
		vec3 caustic = vec3(1.412,1.0,0.0);
		if (!hand){
			filtered = texture(colortex3,texcoord).rgb;
		}
		
		float shading = 1.0 - filtered.b;
	

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
			vec3 projectedShadowPosition = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
			projectedShadowPosition = diagonal3(shadowProjection) * projectedShadowPosition + shadowProjection[3].xyz;
			//apply distortion
			float distortFactor = calcDistort(projectedShadowPosition.xy);
			projectedShadowPosition.xy *= distortFactor;
			//do shadows only if on shadow map
			if (abs(projectedShadowPosition.x) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.y) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.z) < 6.0){
				float rdMul = filtered.x*distortFactor*d0*k/shadowMapResolution;
				const float threshMul = max(2048.0/shadowMapResolution*shadowDistance/128.0,0.95);
				float distortThresh = (sqrt(1.0-NdotLGeom*NdotLGeom)/NdotLGeom+0.7)/distortFactor;
				#ifdef Variable_Penumbra_Shadows
				float diffthresh = distortThresh/6000.0*threshMul;
				#else
				float diffthresh = translucent? 0.0001 : distortThresh/6000.0*threshMul;
				#endif
				#ifdef POM
				#ifdef Depth_Write_POM
					diffthresh += POM_DEPTH/128./4./6.0;
				#endif
				#endif
				projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);
				shading = 0.0;
				for(int i = 0; i < SHADOW_FILTER_SAMPLE_COUNT; i++){
					vec2 offsetS = tapLocation(i,SHADOW_FILTER_SAMPLE_COUNT, 0.0,noise,0.0);

					float weight = 1.0+(i+noise)*rdMul/SHADOW_FILTER_SAMPLE_COUNT*shadowMapResolution;


			#ifdef SHADOWS_ON	
				float isShadow = texture(shadow,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight)));

			#else
				float isShadow = 0;

			#endif	
				//	float isShadow =texture(shadowtex1,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x;

					#ifdef SHADOWS_ON	
						float shadow1    = texture(shadowtex1,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight)));
						float shadow0    = texture(shadowtex0,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight)));
							 shadowCol   = vec4(texture(shadowcolor0,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))));
	

						float transparentshadow = (shadow1-shadow0);	 


						if(shadow0 < 1.0){
							shadowCol = shadowCol * shadow1;
						}
				//	transparentshadow -= shadowCol.a;
						float tempshadow = shadow0;
					if(transparentshadow >0.9) tempshadow = 1-shadowCol.a;
						shading += clamp((tempshadow)/SHADOW_FILTER_SAMPLE_COUNT,0,1);
					#endif	
						if(transparentshadow <=0.0) shadowCol.rgb = vec3(shading);
									
				}
			}
		}
		//custom shading model for translucent objects
		#ifdef Variable_Penumbra_Shadows
		#ifndef LABSSS
		if (translucent) {
			sssAmount = 0.5;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.5*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			diffuseSun *= 1.0 - sssAmount;
			SSS *= sqrt(lightmap.y);
		}

		if (translucent2) {
			sssAmount = 0.2;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			diffuseSun *= 1.0 - sssAmount;
			SSS *= sqrt(lightmap.y);
		}
		
		#else
			if (labsss > 0.0) {
			sssAmount = labsss;
			vec3 extinction = 1.0 - albedo*0.85;
			// Should be somewhat energy conserving
			SSS = exp(-filtered.y*11.0*extinction) + 3.0*exp(-filtered.y*11./3.*extinction);
			float scattering = clamp((0.7+0.3*pi*phaseg(dot(np3, WsunVec),0.85))*1.26*0.25*sssAmount,0.0,1.0);
			SSS *= scattering;
			diffuseSun *= 1.0 - sssAmount;
			SSS *= clamp(sqrt(lightmap.y*2-1.5),0,1);
			SSS *= shading;
		}	
		#endif
		#endif


		if ((diffuseSun*shading > 0.001 || abs(filtered.y-0.1) < 0.0004)){

		if(!hand){	
		#ifdef SCREENSPACE_CONTACT_SHADOWS
				vec3 vector = lightCol.a*sunVec;
				float screenShadow = rayTraceShadow(vector,fragpos,noise);
				shading = min(screenShadow, shading);
				// Out of shadow map
				if (abs(filtered.y-0.1) < 0.0004)
					SSS *= 1;
			#endif 
			}
		

		#ifdef CAVE_LIGHT_LEAK_FIX
			shading = mix(0.0, shading, clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0))*lightmap.y;
			shadowCol.rgb = mix(vec3(0.0), shadowCol.rgb, clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0))*lightmap.y;
		#endif
		}
		#ifdef CLOUDS_SHADOWS
			vec3 pos = p3 + cameraPosition;
			const int rayMarchSteps = 6;
			float cloudShadow = 0.0;
			for (int i = 0; i < rayMarchSteps; i++){
				vec3 cloudPos = pos + WsunVec/abs(WsunVec.y)*(1500+(noise+i)/rayMarchSteps*1700-pos.y);
				cloudShadow += getCloudDensity(cloudPos, 0);
			}
			cloudShadow = mix(1.0,exp(-cloudShadow*cloudDensity*1700/rayMarchSteps),mix(CLOUDS_SHADOWS_STRENGTH,1.0,rainStrength));
			shading *= cloudShadow;
			SSS *= cloudShadow;
		#endif
		





			//combine all light sources
			float shadowmask = luma(shadowCol.rgb * shadowCol.a);
			if( shadowmask > 0.8 && shadowmask < 0.84) shadowCol.rgb = mix(vec3(1.0), shadowCol.rgb,1-shadowCol.a);			
			shadowCol.rgb = clamp(shadowCol.rgb * (1.0 - shading) + shading, vec3(0.0), vec3(1.0));

			shadowCol.rgb = (shadowCol.rgb * diffuseSun + SSS ) ;
			gl_FragData[1].rg =  moment(ivec2(floor(texcoord * vec2(viewWidth, viewHeight)))) ;
			gl_FragData[0] =  vec4(shadowCol.rgb, encodeVec2v2(filtered.yz)) ;




	
	
		

	
	}	


/* RENDERTARGETS: 3*/
}
