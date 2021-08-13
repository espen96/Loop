
#define PCF
#define gbuffer


     uniform int renderStage; 

// 0 Undefined
// 1  Sky
// 2  Sunset and sunrise overlay
// 3  Custom sky
// 4  Sun
// 5  Moon
// 6  Stars
// 7  Void
// 8  Terrain solid
// 9  Terrain cutout mipped
// 10 Terrain cutout
// 11 Entities
// 12 Block entities
// 13 Destroy overlay
// 14 Selection outline
// 15 Debug renderers
// 16 Solid handheld objects
// 17 Terrain translucent
// 18 Tripwire string
// 19 Particles
// 20 Clouds
// 21 Rain and snow
// 22 World border
// 23 Translucent handheld objects
uniform float frameTimeCounter;
varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;
uniform sampler2D normals;
uniform sampler2D specular;
uniform sampler2D texture;
uniform sampler2D noisetex;
#ifdef SHADOWS_ON
uniform sampler2DShadow shadow;
#endif
uniform sampler2D gaux1;
uniform vec4 lightCol;
uniform vec3 sunVec;
uniform vec3 upVec;
uniform float lightSign;						
uniform float frameCounter;
uniform vec2 texelSize;
uniform float skyIntensityNight;
uniform float skyIntensity;
uniform float sunElevation;
uniform float rainStrength;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

#include "/lib/Shadow_Params.glsl"
#include "/lib/noise.glsl"
#include "/lib/kernel.glsl"
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
    float a = sqrt( 1. - x ) * ( -0.16882 * x + 1.56734 );
    return sx > 0. ? a : 3.14159265359 - a;
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

float w0(float a)
{
    return (0.1666)*(a*(a*(-a + 3.0) - 3.0) + 1.0);
}

float w1(float a)
{
    return (0.1666)*(a*a*(3.0*a - 6.0) + 4.0);
}

float w2(float a)
{
    return (0.1666)*(a*(a*(-3.0*a + 3.0) + 3.0) + 1.0);
}

float w3(float a)
{
    return (0.1666)*(a*a*a);
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

float shadow2D_bicubic(sampler2DShadow tex, vec3 sc)
{
	vec2 uv = sc.xy*shadowMapResolution;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );

    float g0x = g0(fuv.x);
    float g1x = g1(fuv.x);
    float h0x = h0(fuv.x);
    float h1x = h1(fuv.x);
    float h0y = h0(fuv.y);
    float h1y = h1(fuv.y);

	vec2 p0 = vec2(iuv.x + h0x, iuv.y + h0y)/shadowMapResolution - 0.5/shadowMapResolution;
	vec2 p1 = vec2(iuv.x + h1x, iuv.y + h0y)/shadowMapResolution - 0.5/shadowMapResolution;
	vec2 p2 = vec2(iuv.x + h0x, iuv.y + h1y)/shadowMapResolution - 0.5/shadowMapResolution;
	vec2 p3 = vec2(iuv.x + h1x, iuv.y + h1y)/shadowMapResolution - 0.5/shadowMapResolution;

    return g0(fuv.y) * (g0x * shadow2D(tex, vec3(p0,sc.z)).x  +
                        g1x * shadow2D(tex, vec3(p1,sc.z)).x) +
           g1(fuv.y) * (g0x * shadow2D(tex, vec3(p2,sc.z)).x  +
                        g1x * shadow2D(tex, vec3(p3,sc.z)).x);
}
float luma(vec3 color) {
	return dot(color,vec3(0.299, 0.587, 0.114));
}
mat3 cotangent( vec3 N, vec3 p, vec2 uv )
{

    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );
 
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
 

    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}



//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* RENDERTARGETS: 2,10,7 */
void main() {
if(renderStage == 18)gl_FragData[3].r = 0.3;
if(renderStage == 19)gl_FragData[3].g = 0.3;
if(renderStage == 17)gl_FragData[3].b = 0.3;
                     gl_FragData[3].a = 1.0;

	gl_FragData[0] = textureLod(texture, lmtexcoord.xy,0)*color;
	vec2 tempOffset=offsets[framemod8];
#ifdef textured
//	float avgBlockLum = luma(textureLod(texture, lmtexcoord.xy,128).rgb*color.rgb);
//	gl_FragData[0].rgb = clamp((gl_FragData[0].rgb)*pow(avgBlockLum,-0.33)*0.85,0.0,1.0);
	gl_FragData[0].rgb = clamp((gl_FragData[0].rgb),0.0,1.0);
#endif	
	
	
	vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize/RENDER_SCALE,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));

	vec2 lm = lmtexcoord.zw;
	vec3 normal = normalMat.xyz;
	vec3 normalTex = textureLod(normals, lmtexcoord.xy , 0).rgb;
	lm *= normalTex.b;
    normalTex = normalTex * 255./127. - 128./127.;
	
    normalTex.z = sqrt( 1.0 - dot( normalTex.xy, normalTex.xy ) );
    normalTex.y = -normalTex.y;
    normalTex.x = -normalTex.x;

    mat3 TBN = cotangent( normal, -fragpos, lmtexcoord.xy );
//    normal = normalize( TBN * clamp(normalTex,-1,1) );	


//	if (gl_FragData[0].a>0.1){




#if defined(glint) || defined(beacon) || defined(spidereyes)
	gl_FragData[0] = textureLod(texture, lmtexcoord.xy,0);

	vec3 albedo = toLinear(gl_FragData[0].rgb*color.rgb);

	float exposure = texelFetch2D(gaux1,ivec2(10,37),0).r;

	vec3 col = albedo*exp(-exposure*3.);


	gl_FragData[0].rgb = col*color.a;
	gl_FragData[0].a = gl_FragData[0].a*0.1;

#else
		vec3 albedo = toLinear(gl_FragData[0].rgb);

	



		
		#ifndef textured		
			float NdotL = lightCol.a*dot(normal,sunVec);
			float NdotU = dot(upVec,normal);
			float diffuseSun = clamp(NdotL,0.0f,1.0f);
		#else
			float NdotL = -lightSign*dot(normal,sunVec);
			float NdotU = dot(upVec,normal);
			float diffuseSun = 0.712;
		#endif			
		
		vec3 direct = texelFetch2D(gaux1,ivec2(6,37),0).rgb/3.1415;


		//compute shadows only if not backface
		if (diffuseSun > 0.001) {
			vec3 p3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;
			vec3 projectedShadowPosition = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
			projectedShadowPosition = diagonal3(shadowProjection) * projectedShadowPosition + shadowProjection[3].xyz;

			//apply distortion
			float distortFactor = calcDistort(projectedShadowPosition.xy);
			projectedShadowPosition.xy *= distortFactor;
			//do shadows only if on shadow map
			if (abs(projectedShadowPosition.x) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.y) < 1.0-1.5/shadowMapResolution){
				const float threshMul = sqrt(2048.0/shadowMapResolution*shadowDistance*0.0078125);
				float distortThresh = 1.0/(distortFactor*distortFactor);
		#ifndef textured		
				float diffthresh = facos(diffuseSun)*distortThresh/800*threshMul;
		#else
		     	float diffthresh = 0.0002;	
		#endif		
				
				projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);

				float noise = interleaved_gradientNoise(tempOffset.x*0.5+0.5);


				vec2 offsetS = vec2(cos( noise*3.14159265359*2.0 ),sin( noise*3.14159265359*2.0 ));

			#ifdef SHADOWS_ON	
				float shading = shadow2D_bicubic(shadow,vec3(projectedShadowPosition + vec3(0.0,0.0,-diffthresh*1.2)));
			#else
				float shading = 0;
			#endif	
				direct *= shading;
			}

		}


		direct *= diffuseSun;

		vec3 ambient = textureLod(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize,0).rgb;

#ifndef textured		
		vec3 diffuseLight = direct + ambient;
#else
	vec3 diffuseLight = direct*lmtexcoord.w + ambient;
#endif

		
		
		
	#if defined(spidereyes)	
	albedo.rgb = toLinear(albedo.rgb)*0.33;
	gl_FragData[0] = albedo;	
	#else	
	#ifdef weather
		gl_FragData[0].rgb = dot(albedo,vec3(1.0))*ambient*10./3.0/150.*0.1;	
	#else

	gl_FragData[0].rgb = diffuseLight*albedo*8.*0.333*0.0066*0.1;	
	#endif
	#endif	
	#ifdef basic
		gl_FragData[0].rgb = diffuseLight*albedo*8.*0.333*0.0066*0.1;	
		gl_FragData[0].a = 1.0;	
	#endif

	gl_FragData[1].rgba = vec4(normal,0);		
#endif		

//	}


	#ifdef SPEC
		gl_FragData[1] = vec4(textureLod(specular, lmtexcoord.xy, 0).rgb,0);
	#else	
		gl_FragData[1] = vec4(0.0);
	#endif	


}
