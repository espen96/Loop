


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

uniform sampler2D texture;
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



uniform vec2 texelSize;
varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;
 uniform int framemod8;






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
    vec2 temp = floor( a * 255. );
	return temp.x*constant1.x+temp.y*constant1.y;
}
float encodeVec2(float x,float y){
    return encodeVec2(vec2(x,y));
}

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)



		const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
									vec2(-1.,3.)/8.,
									vec2(5.0,1.)/8.,
									vec2(-3,-5.)/8.,
									vec2(-5.,5.)/8.,
									vec2(-7.,-1.)/8.,
									vec2(3,7.)/8.,
									vec2(7.,-7.)/8.);
									





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
									
									
									
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* DRAWBUFFERS:137 */
void main() {	


	
vec2 tempOffset=offsets[framemod8];

	
	float noise = interleaved_gradientNoise();
	vec3 normal = normalMat.xyz;
	vec4 specularity = vec4(1.0);
	vec3 mat_data = vec3(0.0);
	vec4 reflected = vec4(0.0);
	vec3 fragC = gl_FragCoord.xyz*vec3(texelSize,1.0);
	vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;

vec3 direct = texelFetch2D(gaux1,ivec2(6,37),0).rgb/3.1415;	
float ao = 1.0;
		
#ifdef POM		
float iswater = normalMat.w;		
float NdotL = lightSign*dot(normal,sunVec);
float NdotU = dot(upVec,normal);

float diffuseSun = clamp(NdotL,0.0f,1.0f);	
float shading = 1.0;
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
				const float threshMul = max(2048.0/shadowMapResolution*shadowDistance/128.0,0.95);
				float distortThresh = (sqrt(1.0-diffuseSun*diffuseSun)/diffuseSun+0.7)/distortFactor;
				float diffthresh = distortThresh/6000.0*threshMul;

				projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);

				shading = 0.0;
				float noise = R2_dither();
				float rdMul = 3.0*distortFactor*d0*k/shadowMapResolution;
				mat2 noiseM = mat2( cos( noise*3.14159265359*2.0 ), -sin( noise*3.14159265359*2.0 ),
									 sin( noise*3.14159265359*2.0 ), cos( noise*3.14159265359*2.0 )
									);
				for(int i = 0; i < 6; i++){
					vec2 offsetS = noiseM*shadowOffsets[i];

					float weight = 1.0+(i+noise)*rdMul/8.0*shadowMapResolution;
					shading += shadow2D(shadow,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x/6.0;
					}



				direct *= shading;
			}

		}	
		direct *= (iswater > 0.9 ? 0.2: 1.0)*diffuseSun*lmtexcoord.w;
		
		
        ao = texture2D(normals, lmtexcoord.xy).z*2.0-1.;
		#endif
		vec3 diffuseLight = direct + texture2D(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize).rgb;
		vec3 color = (color.rgb*ao)*(clamp(clamp(diffuseLight,0,1),0,1));	
	
	
	
	
	
	#ifdef MC_NORMAL_MAP
		vec3 tangent2 = normalize(cross(tangent.rgb,normal)*tangent.w);
		mat3 tbnMatrix = mat3(tangent.x, tangent2.x, normal.x,
								  tangent.y, tangent2.y, normal.y,
						     	  tangent.z, tangent2.z, normal.z);
	#endif

#ifdef POM

		vec2 adjustedTexCoord = fract(vtexcoord.st)*vtexcoordam.pq+vtexcoordam.st;
		vec3 viewVector = normalize(tbnMatrix*fragpos);
		float dist = length(fragpos);
		
		
		
#ifdef POM2		
if (dist < MAX_OCCLUSION_DISTANCE) {
	#ifndef USE_LUMINANCE_AS_HEIGHTMAP
		if ( viewVector.z < 0.0 && readNormal(vtexcoord.st).a < 0.9999 && readNormal(vtexcoord.st).a > 0.00001)
	{
		vec3 interval = viewVector.xyz * intervalMult;
		vec3 coord = vec3(vtexcoord.st, 1.0);
		coord += noise*interval;

		for (int loopCount = 0;
				(loopCount < MAX_OCCLUSION_POINTS) && (readNormal(coord.st).a < coord.p) &&coord.p >= 0.0;
				++loopCount) {
			coord = coord+interval;

		}
		if (coord.t < mincoord) {
			if (readTexture(vec2(coord.s,mincoord)).a == 0.0) {
				coord.t = mincoord;
				discard;
			}
		}
		adjustedTexCoord = mix(fract(coord.st)*vtexcoordam.pq+vtexcoordam.st , adjustedTexCoord , max(dist-MIX_OCCLUSION_DISTANCE,0.0)/(MAX_OCCLUSION_DISTANCE-MIX_OCCLUSION_DISTANCE));
	}



	#endif	
	
	}	
	
#endif	
	
	
	vec3 normal2 = vec3(texture2DGradARB(normals,adjustedTexCoord.xy,dcdx,dcdy).rgb)*2.0-1.;
	float normalCheck = normal2.x + normal2.y;
    if (normalCheck > -1.999){
        if (length(normal2.xy) > 1.0) normal2.xy = normalize(normal2.xy);
        normal2.z = sqrt(1.0 - dot(normal2.xy, normal2.xy));
        normal2 = normalize(clamp(normal2, vec3(-1.0), vec3(1.0)));
    }else{
        normal2 = vec3(0.0, 0.0, 1.0);


    }
	
	
	
	normal = applyBump(tbnMatrix,normal2);
	
	vec4 alb = texture2DGradARB(texture, adjustedTexCoord.xy,dcdx,dcdy);
	
	

	
	
	specularity = texture2DGradARB(specular, adjustedTexCoord, dcdx, dcdy).rgba;



		bool is_metal = false;

		mat_data = labpbr(specularity, is_metal);




 

		float roughness = mat_data.x;

		float emissive = 0.0;
		float F0 = mat_data.y;
		float f0 = (iswater > 0.1?  0.02 : 0.05*(1.0-gl_FragData[0].a))*F0;
		vec3 reflectedVector = reflect(normalize(fragpos), normal);
		float normalDotEye = dot(normal, normalize(fragpos));
		float fresnel = pow(clamp(1.0 + normalDotEye,0.0,1.0), 5.0);
		fresnel = mix(f0,1.0,fresnel);




		vec3 wrefl = mat3(gbufferModelViewInverse)*reflectedVector;
		vec4 sky_c = skyCloudsFromTex(wrefl,gaux1)*(1.0-isEyeInWater);
		if(is_metal){sky_c.rgb *= lmtexcoord.w*lmtexcoord.w*255*255/240.0/240.0/150.0*fresnel/3.0;}
		else{sky_c.rgb *= lmtexcoord.w*lmtexcoord.w*255*255/240.0/240.0/150.0*fresnel/15.0;}





		vec4 reflection = vec4(sky_c.rgb,0.);
		#ifdef SPEC_SCREENSPACE_REFLECTIONS
		vec3 rtPos = rayTrace(reflectedVector,fragpos.xyz,R2_dither(), fresnel);
		if (rtPos.z <1.){

		vec4 fragpositionPrev = gbufferProjectionInverse * vec4(rtPos*2.-1.,1.);
		fragpositionPrev /= fragpositionPrev.w;

		vec3 sampleP = fragpositionPrev.xyz;
		fragpositionPrev = gbufferModelViewInverse * fragpositionPrev;



		vec4 previousPosition = fragpositionPrev + vec4(cameraPosition-previousCameraPosition,0.);
		previousPosition = gbufferPreviousModelView * previousPosition;
		previousPosition = gbufferPreviousProjection * previousPosition;
		previousPosition.xy = previousPosition.xy/previousPosition.w*0.5+0.5;
		if(is_metal){reflection.a = clamp(F0+(1-roughness),0,0.25);
		}else{reflection.a = clamp(F0+(1-roughness),0,0.01);}
		

	    reflection.rgb = texture2D(gaux2,previousPosition.xy).rgb;
		if (reflection.b <= 0.25) reflection.rgb = sky_c.rgb;
		}
		#endif
		reflection.rgb = mix(sky_c.rgb*0.25, reflection.rgb, reflection.a);


			float sunSpec = GGX(normal,normalize(fragpos),  lightSign*sunVec, mat_data.xy)* luma(texelFetch2D(gaux1,ivec2(6,6),0).rgb)*8.0/3./150.0/3.1415 * (1.0-rainStrength*0.9);



		vec3 sp = (reflection.rgb*fresnel+shading*sunSpec);


		if (is_metal) {
			reflected.rgb *= alb.rgb * 0.5 + 0.5;
			reflected.rgb += sp * alb.rgb ;
		} else {
			reflected.rgb += sp ;
		}

		
vec4 data0 = texture2DGradARB(texture, adjustedTexCoord.xy,dcdx,dcdy);

  data0.a = texture2DGradARB(texture, adjustedTexCoord.xy,vec2(0.),vec2(0.0)).a;
	if (data0.a > 0.1) data0.a = normalMat.a*0.5+0.49999;
  else data0.a = 0.0;





	data0.rgb*=color.rgb;
	
	vec4 data1 = clamp(noise*exp2(-8.)+encode(normal),0.,1.0);

	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));
	gl_FragData[1] = vec4(reflected.rgb,0);
	gl_FragData[2].rgb = vec3(0,mat_data.z,0);
	#else


	
	
	
	vec4 data0 = texture2D(texture, lmtexcoord.xy);

  data0.rgb*=color.rgb;
  data0.rgb = toLinear(data0.rgb);
  float avgBlockLum = dot(toLinear(texture2D(texture, lmtexcoord.xy,128).rgb),vec3(0.333));
  data0.rgb = pow(clamp(mix(data0.rgb*1.7,data0.rgb*0.8,avgBlockLum),0.0,1.0),vec3(1.0/2.2333));
  


  #ifdef DISABLE_ALPHA_MIPMAPS
  data0.a = texture2DLod(texture,lmtexcoord.xy,0).a;
  #endif


	if (data0.a > 0.1) data0.a = normalMat.a*0.5+0.49999;
  else data0.a = 0.0;
	#ifdef MC_NORMAL_MAP
	vec3 normal2 = vec3(texture2D(normals, lmtexcoord.xy).rgb)*2.0-1.;
		
	float normalCheck = normal2.x + normal2.y;
    if (normalCheck > -1.999){
        if (length(normal2.xy) > 1.0) normal2.xy = normalize(normal2.xy);
        normal2.z = sqrt(1.0 - dot(normal2.xy, normal2.xy));
        normal2 = normalize(clamp(normal2, vec3(-1.0), vec3(1.0)));
    }else{
        normal2 = vec3(0.0, 0.0, 1.0);

    }

	normal = applyBump(tbnMatrix,normal2);
	#endif
	vec4 data1 = clamp(noise/256.+encode(normal),0.,1.0);

	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));
	#endif

	gl_FragData[1] = clamp(vec4(reflected.rgb,0)*1,0.0,10.0);
	gl_FragData[2].rgb = vec3(0,mat_data.z,0);

}
