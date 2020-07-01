#version 120
#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"




varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;



uniform sampler2D texture;
uniform sampler2DShadow shadow;
uniform sampler2D gaux1;
uniform vec3 sunVec;
uniform vec3 upVec;
uniform float lightSign;

uniform vec2 texelSize;
uniform float skyIntensityNight;
uniform float skyIntensity;
uniform float sunElevation;
uniform float rainStrength;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
#include "/lib/Shadow_Params.glsl"
#include "/lib/util2.glsl"
//faster and actually more precise than pow 2.2
vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}

#ifdef PCF
const vec2 shadowOffsets[4] = vec2[4](vec2( 0.1250,  0.0000 ),
vec2( -0.1768, -0.1768 ),
vec2( -0.0000,  0.3750 ),
vec2(  0.3536, -0.3536 )
);
#endif
float facos(float sx){
    float x = clamp(abs( sx ),0.,1.);
    return sqrt( 1. - x ) * ( -0.16882 * x + 1.56734 );
}

vec2 tapLocation(int sampleNumber, float spinAngle,int nb, float nbRot)
{
	float startJitter = (spinAngle/6.28);
    float alpha = sqrt(sampleNumber + startJitter/nb );
    float angle = alpha * (nbRot * 6.28) + spinAngle*2.;

    float ssR = alpha;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*ssR;
}
uniform int framemod8;
uniform int framecouter;
		const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
									vec2(-1.,3.)/8.,
									vec2(5.0,1.)/8.,
									vec2(-3,-5.)/8.,
									vec2(-5.,5.)/8.,
									vec2(-7.,-1.)/8.,
									vec2(3,7.)/8.,
									vec2(7.,-7.)/8.);

float w0(float a)
{
    return (1.0/6.0)*(a*(a*(-a + 3.0) - 3.0) + 1.0);
}

float w1(float a)
{
    return (1.0/6.0)*(a*a*(3.0*a - 6.0) + 4.0);
}

float w2(float a)
{
    return (1.0/6.0)*(a*(a*(-3.0*a + 3.0) + 3.0) + 1.0);
}

float w3(float a)
{
    return (1.0/6.0)*(a*a*a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0 + w1(a) / (w0(a) + w1(a));
}

float h1(float a)
{
    return 1.0 + w3(a) / (w2(a) + w3(a));
}



//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* DRAWBUFFERS:2 */
void main() {

	gl_FragData[0] = texture2D(texture, lmtexcoord.xy)*color;
	vec2 tempOffset=offsets[framemod8];

		vec3 albedo = toLinear(gl_FragData[0].rgb);

		vec3 normal = normalMat.xyz;
		vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));

		float NdotL = -lightSign*dot(normal,sunVec);
		float NdotU = dot(upVec,normal);
		float diffuseSun = 0.712;
		vec3 direct = texelFetch2D(gaux1,ivec2(6,37),0).rgb/3.1415;

		//compute shadows only if not backface



		direct *= diffuseSun;

		vec3 ambient = texture2D(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize).rgb;

		vec3 diffuseLight = direct*lmtexcoord.w + ambient;


		gl_FragData[0].rgb = diffuseLight*albedo*8./3.0/150.*0.1;




}
