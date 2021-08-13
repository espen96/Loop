#version 150
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
//#define POM

#include "/lib/res_params.glsl"
#define SSAO

#define ROUGHREF
#define power

#define Depth_Write_POM	// POM adjusts the actual position, so screen space shadows can cast shadows on POM
#define POM_DEPTH 0.25 // [0.025 0.05 0.075 0.1 0.125 0.15 0.20 0.25 0.30 0.50 0.75 1.0] //Increase to increase POM strength
#define CAVE_LIGHT_LEAK_FIX // Hackish way to remove sunlight incorrectly leaking into the caves. Can inacurrately create shadows in some places
//#define CLOUDS_SHADOWS
#define CLOUDS_SHADOWS_STRENGTH 1.0 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]




#define CLOUDS_QUALITY 0.35 //[0.1 0.125 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0]
#define SPEC_SSR_QUALITY 2 //[1 2 3 4 5 6 7 8 9 10 ]


#define TORCH_R 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_G 0.5 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_B 0.2 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]

#define Emissive_Strength 2.00 // [0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20 1.30 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10 3.20 3.30 3.40 3.50 3.60 3.70 3.80 3.90 4.00 4.10 4.20 4.30 4.40 4.50 4.60 4.70 4.80 4.90 5.00 5.10 5.20 5.30 5.40 5.50 5.60 5.70 5.80 5.90 6.00 6.10 6.20 6.30 6.40 6.50 6.60 6.70 6.80 6.90 7.00 7.10 7.20 7.30 7.40 7.50 7.60 7.70 7.80 7.90 8.00 8.10 8.20 8.30 8.40 8.50 8.60 8.70 8.80 8.90 9.00 9.10 9.20 9.30 9.40 9.50 9.60 9.70 9.80 9.90 10.00 15.00 20.00 30.00 50.00 100.00 150.00 200.00]


const bool shadowHardwareFiltering = true;

flat in vec4 lightCol; //main light source color (rgb),used light source(1=sun,-1=moon)
flat in vec3 ambientUp;
flat in vec3 ambientLeft;
flat in vec3 ambientRight;
flat in vec3 ambientB;
flat in vec3 ambientF;
flat in vec3 ambientDown;
flat in vec3 WsunVec;
flat in vec2 TAA_Offset;
flat in float tempOffsets;
flat in vec3 refractedSunVec;

flat in vec4 exposure;
flat in vec2 coord;

uniform sampler2D colortex0;//clouds
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortex2;
uniform sampler2D colortex5;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex9;
uniform sampler2D colortex10;
uniform sampler2D colortex11;
uniform sampler2D colortex12;
uniform sampler2D colortex14;
uniform sampler2D colortex15;
uniform sampler2D colortex6; // Noise
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth
uniform sampler2D colortex13;

//uniform sampler2D shadow;
//uniform sampler2D shadowcolor1;
//uniform sampler2D shadowcolor0;


uniform mat4 shadowProjectionInverse;

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
uniform vec2 viewSize;
uniform vec3 cameraPosition;
uniform vec3 sunVec;
uniform ivec2 eyeBrightnessSmooth;

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
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
#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/volumetricClouds.glsl"
#include "/lib/kernel.glsl"


vec3 screenToViewSpace(vec3 screenpos, mat4 projInv, const bool taaAware) {
     screenpos   = screenpos*2.0-1.0;


vec3 viewpos    = vec3(vec2(projInv[0].x, projInv[1].y)*screenpos.xy + projInv[3].xy, projInv[3].z);
     viewpos    /= projInv[2].w*screenpos.z + projInv[3].w;
    
    return viewpos;
}
vec3 screenToViewSpace(vec3 screenpos, mat4 projInv) {    
    return screenToViewSpace(screenpos, projInv, true);
}

vec3 screenToViewSpace(vec3 screenpos) {
    return screenToViewSpace(screenpos, gbufferProjectionInverse);
}
vec3 screenToViewSpace(vec3 screenpos, const bool taaAware) {
    return screenToViewSpace(screenpos, gbufferProjectionInverse, taaAware);
}	
float sqr(float x) {
    return x*x;
}
float computeVariance(sampler2D tex, ivec2 pos) {
    float sum_msqr  = 0.0;
    float sum_mean  = 0.0;

    for (int i = 0; i<9; i++) {
        ivec2 deltaPos     = kernelO_3x3[i];


        vec3 col    = texelFetch2D(tex, pos + deltaPos, 0).rgb;
        float lum   = luma(col);

        sum_msqr   += sqr(lum);
        sum_mean   += lum;
    }
    sum_msqr /= 9.0;
    sum_mean /= 9.0;

    return abs(sum_msqr - sqr(sum_mean)) * 1/(max(sum_mean, 1e-25));
}
float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}



vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
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
vec2 R2_samples(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}

//#include "/lib/rsm.glsl"
vec3 toScreenSpaceVector(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return normalize(fragposition.xyz);
}





vec4 computeVariance2(sampler2D tex, ivec2 pos) {

  vec2 texcoord = gl_FragCoord.xy*texelSize;
  		vec4 currentPosition = vec4(texcoord.x * 2.0 - 1.0, texcoord.y * 2.0 - 1.0, 2.0 * texture(depthtex1, texcoord.st).x - 1.0, 1.0);

		vec4 fragposition = gbufferProjectionInverse * currentPosition;
		fragposition = gbufferModelViewInverse * fragposition;
		fragposition /= fragposition.w;
		fragposition.xyz += cameraPosition;
		vec4 previousPosition = fragposition;
		previousPosition.xyz -= previousCameraPosition;
		previousPosition = gbufferPreviousModelView * previousPosition;
		previousPosition = gbufferPreviousProjection * previousPosition;
		previousPosition /= previousPosition.w;
		vec2 velocity = (currentPosition - previousPosition).st; 
        float weightSum = 1.0;
        int radius = 3; //  7x7 Gaussian Kernel
        vec2 moment = velocity;
        vec4 c = texelFetch(colortex14, pos, 0);
        float histlen = texelFetch(colortex14, pos, 0).a;
                float depth = texelFetch(depthtex0, pos, 0).x;
                        vec3 normal = texelFetch(colortex10, pos, 0).xyz;

for (int yy = -radius; yy <= radius; ++yy)
{
    for (int xx = -radius; xx <= radius; ++xx)
    {
        //  We already have the center data
        if (xx != 0 && yy != 0) { continue; }

        //  Sample current point data with current uv
        ivec2 p = pos + ivec2(xx, yy);
        vec4 curColor = texelFetch(colortex12, p, 0);
        float curDepth = texelFetch(depthtex0, p, 0).x;
        vec3 curNormal = texelFetch(colortex10, p, 0).xyz;

        //  Determine the average brightness of this sample
        //  Using International Telecommunications Union's ITU BT.601 encoding params
        float l = luma(curColor.rgb);

        float weightDepth = abs(curDepth - depth) / (depth * length(vec2(xx, yy)) + 1.0e-2);
        float weightNormal = pow(max(0, dot(curNormal, normal)), 16.0);

   //     uint curMeshID =  floatBitsToUint(texelFetch(tMeshID, p, 0).r);

   //     float w = exp(-weightDepth) * weightNormal * (meshID == curMeshID ? 1.0 : 0.0);
        float w = exp(-weightDepth) * weightNormal *1;

        if (isnan(w))
            w = 0.0;

        weightSum += w;

        moment += vec2(l, l * l) * w;
        c.rgb += curColor.rgb * w;
    }
}

moment /= weightSum;

c.rgb /= weightSum;
	gl_FragData[5] = vec4(c.rgb,(1.0 + 3.0 * (1.0 - histlen)) * max(0.0, moment.y - moment.x * moment.x));
//varianceSpatial = (1.0 + 2.0 * (1.0 - histlen)) * max(0.0, moment.y - moment.x * moment.x);
return  vec4(c.rgb, (1.0 + 3.0 * (1.0 - histlen)) * max(0.0, moment.y - moment.x * moment.x));

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
#ifdef SSGI
#include "/lib/ssgi.glsl"
#endif






void ssao(inout float occlusion,vec3 fragpos,float mulfov,float dither,vec3 normal, float z)
{

	const float tan70 = tan(70.*3.14/180.);
	float mulfov2 = gbufferProjection[1][1]/tan70;


	float maxR2 = fragpos.z*fragpos.z*mulfov2*2.*1.412/50.0;



	float rd = mulfov2*0.04;	//pre-rotate direction
	
	float n = 0.;

	occlusion = 0.0;
	
	int samples = 6;


	vec2 acc = -vec2(TAA_Offset)*texelSize*0.5;
	float mult = (dot(normal,normalize(fragpos))+1.0)*0.5+0.5;

	vec2 v = fract(vec2(dither,blueNoise()) + (frameCounter%10000) * vec2(0.75487765, 0.56984026));
	for (int j = 0; j < samples ;j++) {

			vec2 sp = tapLocation(j,v.x,7,88.,v.y);
			vec2 sampleOffset = sp*rd;
			ivec2 offset = ivec2(gl_FragCoord.xy + sampleOffset*vec2(viewWidth,viewHeight*aspectRatio)*RENDER_SCALE);
			if (offset.x >= 0 && offset.y >= 0 && offset.x < viewWidth*RENDER_SCALE.x && offset.y < viewHeight*RENDER_SCALE.y ) {
				vec3 t0 = toScreenSpace(vec3(offset*texelSize+acc+0.5*texelSize,texelFetch2D(depthtex1,offset,0).x) * vec3(1.0/RENDER_SCALE, 1.0));

				vec3 vector = t0.xyz - fragpos;
				float dsquared = dot(vector,vector);
				if (dsquared > 1e-5){
					if (dsquared < maxR2){
						float NdotV = clamp(dot(vector*inversesqrt(dsquared), normalize(normal)),0.,1.);
						occlusion += NdotV * clamp(1.0-dsquared/maxR2,0.0,1.0);
					}
					n += 1.0;
				}
			}
		}



		occlusion = clamp(1.0-occlusion/n*1.6,0.,1.0);
		//occlusion = mult;


}


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

vec3 constructNormal(float depthA, vec2 texcoords, sampler2D depthtex) {
    const vec2 offsetB = vec2(0.0,0.001);
    const vec2 offsetC = vec2(0.001,0.0);
  
    float depthB = texture(depthtex, texcoords + offsetB).r;
    float depthC = texture(depthtex, texcoords + offsetC).r;
  
    vec3 A = getDepthPoint(texcoords, depthA);
	vec3 B = getDepthPoint(texcoords + offsetB, depthB);
	vec3 C = getDepthPoint(texcoords + offsetC, depthC);

	vec3 AB = normalize(B - A);
	vec3 AC = normalize(C - A);

	vec3 normal =  -cross(AB, AC);
	// normal.z = -normal.z;

	return normalize(normal);
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

float encodeVec2v2(vec2 a){
    ivec2 bf = ivec2(a*255.);
    return float( bf.x|(bf.y<<8) ) / 65535.;
}


vec3 decode3x16(float a){
    int bf = int(a*65535.);
    return vec3(bf%32, (bf>>5)%64, bf>>11) / vec3(31,63,31);
}

float encode2x16(vec2 a){
    ivec2 bf = ivec2(a*255.);
    return float( bf.x|(bf.y<<8) ) / 65535.;
}

vec2 decode2x16(float a){
    int bf = int(a*65535.);
    return vec2(bf%256, bf>>8) / 255.;
}
float encodeNormal3x16(vec3 a){
    vec3 b  = abs(a);
    vec2 p  = a.xy / (b.x + b.y + b.z);
    vec2 sp = vec2(greaterThanEqual(p, vec2(0.0))) * 2.0 - 1.0;

    vec2 encoded = a.z <= 0.0 ? (1.0 - abs(p.yx)) * sp : p;

    encoded = encoded * 0.5 + 0.5;

    return encode2x16(encoded);
}

vec3 decodeNormal3x16(float encoded){
    vec2 a = decode2x16(encoded);

    a = a * 2.0 - 1.0;
    vec2 b = abs(a);
    float z = 1.0 - b.x - b.y;
    vec2 sa = vec2(greaterThanEqual(a, vec2(0.0))) * 2.0 - 1.0;

    vec3 decoded = normalize(vec3(
        z < 0.0 ? (1.0 - b.yx) * sa : a.xy,
        z
    ));

    return decoded;
}



void main() {
	
	vec2 texcoord = gl_FragCoord.xy*texelSize;
	float z = texture(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	float noise = blueNoise();


	vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos;
	vec3 np3 = normalize(p3);
		vec3 directLightCol = lightCol.rgb;

	if (z <=1.0) {

		p3 += gbufferModelViewInverse[3].xyz;
		vec4 trpData = texture(colortex7,texcoord);
		bool iswater = texture(colortex7,texcoord).a > 0.99;
		vec4 data = texture(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		vec4 transparent = texture(colortex2,texcoord);
		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
//		vec3 normal = mat3(gbufferModelViewInverse) * worldToView(decode(dataUnpacked0.yw));
		vec3 normalorg = texture(colortex10,texcoord).rgb+texture(colortex8,texcoord).rgb;
		vec3 normal2 =  worldToView(decode(dataUnpacked0.yw));		
    	if (normalorg.r >0.9 && normalorg.g >0.9 && normalorg.b > 0.9) normalorg = constructNormal(texture(depthtex0, texcoord.st).r, texcoord, depthtex0);

		gl_FragData[4] = trpData;
		vec3 normal = mat3(gbufferModelViewInverse) * normalorg;
  		gl_FragData[3].rgba = vec4(normalorg.rgb,ld(texture(depthtex0,texcoord).r));		

		bool hand = abs(dataUnpacked1.w-0.75) <0.01;


		vec2 lightmap = dataUnpacked1.yz;

		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;	// Strong translucency
		bool translucent2 = abs(dataUnpacked1.w-0.6) <0.01;	// Weak translucency

		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		
		float NdotLGeom = dot(normal, WsunVec);
		float NdotL = NdotLGeom;
		if ((iswater && isEyeInWater == 0) || (!iswater && isEyeInWater == 1))
			NdotL = dot(normal, refractedSunVec);

		float diffuseSun = clamp(NdotL,0.,1.0);



		float sssAmount = 0.0;

		#ifdef Variable_Penumbra_Shadows
		// compute shadows only if not backfacing the sun
		// or if the blocker search was full or empty
		// always compute all shadows at close range where artifacts may be more visible
		if (diffuseSun > 0.001) {
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
		if (translucent) {
			sssAmount = 0.5;
		}
		if (translucent2) {
			sssAmount = 0.2;
		}
		#endif

		vec3 ambientCoefs = normal/dot(abs(normal),vec3(1.));
		vec3 ambientLight = ambientUp*mix(clamp(ambientCoefs.y,0.,1.), 1.0/6.0, sssAmount);
		vec3 ambientLight2 = vec3(0.0);		
		ambientLight += ambientDown*mix(clamp(-ambientCoefs.y,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientRight*mix(clamp(ambientCoefs.x,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientLeft*mix(clamp(-ambientCoefs.x,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientB*mix(clamp(ambientCoefs.z,0.,1.), 1.0/6.0, sssAmount);
		ambientLight += ambientF*mix(clamp(-ambientCoefs.z,0.,1.), 1.0/6.0, sssAmount);

		vec3 custom_lightmap = texture(colortex4,(lightmap*15.0+0.5+vec2(0.0,19.))*texelSize).rgb*10./150./3.;
		float emitting = 0.0;
		
		float labemissive = texture(colortex10, texcoord).a;
		if (emissive || (hand && heldBlockLightValue > 0.1)){



			custom_lightmap.y = 0.1;
			custom_lightmap.y += labemissive;

		}
			
		vec3 ambientLight3 = ambientLight * custom_lightmap.x + custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*vec3(TORCH_R,TORCH_G,TORCH_B);		

			#ifdef SSGI

				
				     ambientLight2 = rtGI(normal,normalorg, blueNoise(gl_FragCoord.xy), fragpos,sssAmount, ambientLight* custom_lightmap.x, custom_lightmap.z*vec3(0.9,1.0,1.5) + custom_lightmap.y*(vec3(TORCH_R,TORCH_G,TORCH_B)), normalize(albedo+1e-5)*0.7,ld(z),lightmap.xy, emissive, hand, texcoord);
		
			if(hand) ambientLight2 = ambientLight3;
		
			#else
					ambientLight2 = ambientLight3;
			#endif


		

	vec4 historyGData    = vec4(1.0);
	vec4 indirectHistory = vec4(ambientLight2,0);
	vec3 indirectCurrent = ambientLight2;
	#ifdef SSGI
	#ifdef ssgi_temporal


		if(!hand)	temporal( indirectCurrent, historyGData, indirectHistory, fragpos, normalorg,  z, texcoord ,  hand, ambientLight3, lightmap);




	gl_FragData[2] = historyGData;
	gl_FragData[1] = indirectHistory;

	#endif
	#endif

			#ifndef SSGI

				float ao = 1.0;
				if (!hand)
					ssao(ao,fragpos,1.0,noise,normalorg,z);
				ambientLight2 *= ao;
				indirectCurrent *= ao;
							
		
			#endif	

	gl_FragData[0].rgba = vec4(indirectCurrent,texture(colortex10,texcoord).a);
gl_FragData[4] = vec4(1.0);
	}		



/* RENDERTARGETS: 8,12,9,10,15,11 */
}
