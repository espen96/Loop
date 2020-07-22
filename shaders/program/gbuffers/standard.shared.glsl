#define SPEC

#ifndef USE_LUMINANCE_AS_HEIGHTMAP
#ifndef MC_NORMAL_MAP
#undef POM
#endif
#endif

#ifdef POM
#define MC_NORMAL_MAP
#endif



const float mincoord = 1.0/4096.0;
const float maxcoord = 1.0-mincoord;

#ifdef MC_NORMAL_MAP
varying vec4 tangent;
uniform float wetness;
uniform sampler2D normals;
#endif

varying vec4 vtexcoordam; // .st for add, .pq for mul
varying vec4 vtexcoord;
vec2 dcdx = dFdx(vtexcoord.st*vtexcoordam.pq);
vec2 dcdy = dFdy(vtexcoord.st*vtexcoordam.pq);
uniform sampler2DShadow shadow;

uniform sampler2D gaux2;
uniform sampler2D gaux1;
uniform sampler2D depthtex1;

uniform vec4 lightCol;
uniform vec3 sunVec;
uniform float frameTimeCounter;
uniform float lightSign;
uniform float near;
uniform float far;
uniform float moonIntensity;
uniform float sunIntensity;
uniform vec3 sunColor;
uniform vec3 nsunColor;
uniform vec3 upVec;
uniform float sunElevation;
uniform float fogAmount;

uniform float rainStrength;
uniform float skyIntensityNight;
uniform float skyIntensity;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;

uniform int frameCounter;
uniform int isEyeInWater;
uniform vec2 texelSize;
varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;
uniform int framemod8;	

uniform sampler2D noisetex;


#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/projections.glsl"

#include "/lib/waterBump.glsl"
#include "/lib/clouds.glsl"
#include "/lib/stars.glsl"
#include "/lib/util2.glsl"

uniform sampler2D specular;
#ifdef POM2

const vec3 intervalMult = vec3(1.0, 1.0, 1.0/POM_DEPTH)/POM_MAP_RES * 1.0;
const float MAX_OCCLUSION_DISTANCE = MAX_DIST;
const float MIX_OCCLUSION_DISTANCE = MAX_DIST*0.9;
const int   MAX_OCCLUSION_POINTS   = MAX_ITERATIONS;
#endif
#ifdef POM

#endif


uniform sampler2D texture;


uniform vec4 entityColor;



#include "/lib/sky_gradient.glsl"




varying vec3 viewVector;


float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+frameTimeCounter*51.9521);
}

float pow2(float x) {
    return x*x;
}					 

//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 unenc)
{    
	unenc.xy = unenc.xy / dot(abs(unenc), vec3(1.0)) + 0.00390625;
	unenc.xy = unenc.z <= 0.0 ? (1.0 - abs(unenc.yx)) * sign(unenc.xy) : unenc.xy;
    vec2 encn = unenc.xy * 0.5 + 0.5;
	
    return vec4((encn),vec2(lmtexcoord.z,lmtexcoord.w));
}

#ifdef MC_NORMAL_MAP
vec3 applyBump(mat3 tbnMatrix, vec3 bump)
{

		float bumpmult = BUMP_MULT-wetness*0.95;

		bump = bump * vec3(bumpmult, bumpmult, bumpmult) + vec3(0.0f, 0.0f, 1.0f - bumpmult);

		return normalize(bump*tbnMatrix);
}
#endif

//encoding by jodie
float encodeVec2(vec2 a){
    const vec2 constant1 = vec2( 1., 256.) / 65535.;
    vec2 temp = floor( a * 254. );
	return temp.x*constant1.x+temp.y*constant1.y;
}
float encodeVec2(float x,float y){
    return encodeVec2(vec2(x,y));
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)




									





#ifdef POM
vec4 readNormal(in vec2 coord)
{
	return texture2DGradARB(normals,fract(coord)*vtexcoordam.pq+vtexcoordam.st,dcdx,dcdy);
}
vec4 readTexture(in vec2 coord)
{
	return texture2DGradARB(texture,fract(coord)*vtexcoordam.pq+vtexcoordam.st,dcdx,dcdy);
}
		
									
float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
}
float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}
vec3 nvec3(vec4 pos){
    return pos.xyz/pos.w;
}
float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}
vec4 nvec4(vec3 pos){
    return vec4(pos.xyz, 1.0);
}
vec3 rayTrace(vec3 dir,vec3 position,float dither, float fresnel){

    float quality = mix(10,SSR_STEPS,fresnel);
    vec3 clipPosition = toClipSpace3(position);
	float rayLength = ((position.z + dir.z * far*sqrt(3.)) > -near) ?
       (-near -position.z) / dir.z : far*sqrt(3.);
    vec3 direction = normalize(toClipSpace3(position+dir*rayLength)-clipPosition);  //convert to clip space
    direction.xy = normalize(direction.xy);

    //get at which length the ray intersects with the edge of the screen
    vec3 maxLengths = (step(0.,direction)-clipPosition) / direction;
    float mult = min(min(maxLengths.x,maxLengths.y),maxLengths.z);


    vec3 stepv = direction * mult / quality;




	vec3 spos = clipPosition + stepv*dither;
	float minZ = clipPosition.z;
	float maxZ = spos.z+stepv.z*0.5;
	spos.xy+=offsets[framemod8]*texelSize*0.5;
	//raymarch on a quarter res depth buffer for improved cache coherency


    for (int i = 0; i < int(quality+1); i++) {

			float sp=texelFetch2D(depthtex1,ivec2(spos.xy/texelSize),0).x;

            if(sp <= max(maxZ,minZ) && sp >= min(maxZ,minZ)){
							return vec3(spos.xy,sp);

	        }
        spos += stepv;
		//small bias
		minZ = maxZ-0.00004/ld(spos.z);
		maxZ += stepv.z;
    }

    return vec3(1.1);
}






float linStep(float x, float low, float high) {
    float t = clamp(((x-low)/(high-low)),0,1);
    return t;
}



float facos(float sx){
    float x = clamp(abs( sx ),0.,1.);
    float a = sqrt( 1. - x ) * ( -0.16882 * x + 1.56734 );
    return sx > 0. ? a : pi - a;
}




	float bayer2(vec2 a){
	a = floor(a);
    return fract(dot(a,vec2(0.5,a.y*0.75)));
}									
									
float cdist(vec2 coord) {
	return max(abs(coord.s-0.5),abs(coord.t-0.5))*2.0;
}									
									
									
									
#define PW_DEPTH 1.0 //[0.5 1.0 1.5 2.0 2.5 3.0]
#define PW_POINTS 1 //[2 4 6 8 16 32]


#define bayer4(a)   (bayer2( .5*(a))*.25+bayer2(a))
#define bayer8(a)   (bayer4( .5*(a))*.25+bayer2(a))
#define bayer16(a)  (bayer8( .5*(a))*.25+bayer2(a))
#define bayer32(a)  (bayer16(.5*(a))*.25+bayer2(a))
#define bayer64(a)  (bayer32(.5*(a))*.25+bayer2(a))
#define bayer128(a) fract(bayer64(.5*(a))*.25+bayer2(a))									
									
vec3 getParallaxDisplacement(vec3 posxz, float iswater,float bumpmult,vec3 viewVec) {
	float waveZ = mix(20.0,0.25,iswater);
	float waveM = mix(0.0,4.0,iswater);

	vec3 parallaxPos = posxz;
	vec2 vec = viewVector.xy * (1.0 / float(PW_POINTS)) * 22.0 * PW_DEPTH;
	float waterHeight = getWaterHeightmap(posxz.xz, waveM, waveZ, iswater) * 0.5;
parallaxPos.xz += waterHeight * vec;

	return parallaxPos;

}									
vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter,float distort)
{
    float alpha = (sampleNumber+jitter)/nb;
    float angle = jitter*6.28 + alpha * nbRot * 6.28;

    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*sqrt(alpha);
}
									
									
									
float GGX (vec3 n, vec3 v, vec3 l, vec2 material) {
    float F0  = material.y;
    float r = pow2(material.x);


  vec3 h = l - v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = (dot(h,l)*hn);
  float dotNH = (dot(h,n)*hn);
  float dotNL = (dot(n,l));

  float denom = dotNH * r - dotNH + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  float F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}





float R2_dither(){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0/1.6180339887 * frameCounter);
}									
									
const vec2 shadowOffsets[6] = vec2[6](vec2(  0.5303,  0.5303 ),
vec2( -0.6250, -0.0000 ),
vec2(  0.3536, -0.3536 ),
vec2( -0.0000,  0.3750 ),
vec2( -0.1768, -0.1768 ),
vec2( 0.1250,  0.0000 ));									
									
									
vec3 MetalCol(float f0){
    int metalidx = int(f0 * 255.0);

    if (metalidx == 230) return vec3(0.24867, 0.22965, 0.21366); //iron
    if (metalidx == 231) return vec3(0.88140, 0.57256, 0.11450); //gold
    if (metalidx == 232) return vec3(0.81715, 0.82021, 0.83177); //aluminium
    if (metalidx == 233) return vec3(0.27446, 0.27330, 0.27357); //chrome
    if (metalidx == 234) return vec3(0.84430, 0.48677, 0.22164); //copper
    if (metalidx == 235) return vec3(0.36501, 0.35675, 0.37653); //lead
    if (metalidx == 236) return vec3(0.42648, 0.37772, 0.31138); //platinum
    if (metalidx == 237) return vec3(0.91830, 0.89219, 0.83662); //silver
    return vec3(1.0);
}									
									
vec3 labpbr(vec4 unpacked_tex, out bool is_metal) {
	vec3 mat_data = vec3(1.0, 0.0, 0.0);

    mat_data.x  = pow2(1.0 - unpacked_tex.x);   //roughness
    mat_data.y  = (unpacked_tex.y);         //f0

    unpacked_tex.w = unpacked_tex.w * 255.0;

    mat_data.z  = unpacked_tex.w < 254.5 ? linStep(unpacked_tex.w, 0.0, 254.0) : 0.0; //emission

    is_metal    = (unpacked_tex.y * 255.0) > 229.5;


	return mat_data;
}	


								
#endif								 
		const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
									vec2(-1.,3.)/8.,
									vec2(5.0,1.)/8.,
									vec2(-3,-5.)/8.,
									vec2(-5.,5.)/8.,
									vec2(-7.,-1.)/8.,
									vec2(3,7.)/8.,
									vec2(7.,-7.)/8.);	