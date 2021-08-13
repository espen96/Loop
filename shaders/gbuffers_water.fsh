
#define gbuffer
varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;
uniform sampler2D normals;
varying vec3 binormal;
varying vec3 tangent;
varying vec3 viewVector;
varying float dist;
varying float lumaboost;
#include "/lib/res_params.glsl"
#define SCREENSPACE_REFLECTIONS	//can be really expensive at high resolutions/render quality, especially on ice
#define SSR_STEPS 10 //[10 15 20 25 30 35 40 50 100 200 400]
#define SUN_MICROFACET_SPECULAR // If enabled will use realistic rough microfacet model, else will just reflect the sun. No performance impact.
#define USE_QUARTER_RES_DEPTH // Uses a quarter resolution depth buffer to raymarch screen space reflections, improves performance but may introduce artifacts
#define saturate(x) clamp(x,0.0,1.0)

uniform sampler2D texture;
uniform sampler2D noisetex;

#ifdef SHADOWS_ON
uniform sampler2DShadow shadow;
#endif

uniform sampler2D gaux2;
uniform sampler2D gaux3;
uniform sampler2D gaux1;
uniform sampler2D depthtex1;

uniform vec4 lightCol;
uniform vec3 sunVec;
uniform float frameTimeCounter;
uniform float waveScale;
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
uniform vec2 texelSize;
uniform float rainStrength;
uniform float skyIntensityNight;
uniform float skyIntensity;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;
uniform int framemod8;
uniform int frameCounter;
uniform int isEyeInWater;
#include "/lib/noise.glsl"
#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/projections.glsl"
#include "/lib/sky_gradient.glsl"
#include "/lib/waterBump.glsl"
#include "/lib/kernel.glsl"




float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
}
float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
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


    vec3 stepv = direction * mult / quality*vec3(RENDER_SCALE,1.0);




	vec3 spos = clipPosition*vec3(RENDER_SCALE,1.0) + stepv*dither;
	float minZ = clipPosition.z;
	float maxZ = spos.z+stepv.z*0.5;
	spos.xy += offsets[framemod8]*texelSize*0.5/RENDER_SCALE;

    for (int i = 0; i <= int(quality); i++) {
			#ifdef USE_QUARTER_RES_DEPTH
			// decode depth buffer
			float sp = sqrt(texelFetch2D(gaux1,ivec2(spos.xy/texelSize*0.25),0).w/65000.0);
			sp = invLinZ(sp);
          if(sp <= max(maxZ,minZ) && sp >= min(maxZ,minZ)){
						return vec3(spos.xy/RENDER_SCALE,sp);
	        }
        spos += stepv;
			#else
			float sp = texelFetch2D(depthtex1,ivec2(spos.xy/texelSize),0).r;
          if(sp <= max(maxZ,minZ) && sp >= min(maxZ,minZ)){
						return vec3(spos.xy/RENDER_SCALE,sp);
	        }
        spos += stepv;
			#endif
		//small bias
		minZ = maxZ-0.00004/ld(spos.z);
		maxZ += stepv.z;
    }

    return vec3(1.1);
}




	#define PW_DEPTH 1.0 //[0.5 1.0 1.5 2.0 2.5 3.0]
	#define PW_POINTS 1 //[2 4 6 8 16 32]


vec3 getParallaxDisplacement(vec3 posxz, float iswater,float bumpmult,vec3 viewVec) {
	float waveZ = mix(20.0,0.25,iswater);
	float waveM = mix(0.0,4.0,iswater);

	vec3 parallaxPos = posxz;
	vec2 vec = viewVector.xy * (1.0 / float(PW_POINTS)) * PW_DEPTH;
	float waterHeight = getWaterHeightmap(posxz.xz, iswater) * 2.0;
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

float GGX (vec3 n, vec3 v, vec3 l, float r, float F0) {
  r*=r;r*=r;

  vec3 h = l + v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = clamp(dot(h,l)*hn,0.,1.);
  float dotNH = clamp(dot(h,n)*hn,0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNHsq = dotNH*dotNH;

  float denom = dotNHsq * r - dotNHsq + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  float F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}
float Pow5(float x) { float x2 = x * x; return x2 * x2 * x; }
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

/* RENDERTARGETS: 2,7 */
void main() {
	if (gl_FragCoord.x * texelSize.x < RENDER_SCALE.x  && gl_FragCoord.y * texelSize.y < RENDER_SCALE.y )	{
		vec2 tempOffset=offsets[framemod8];
		float iswater = normalMat.w;
		vec3 fragC = gl_FragCoord.xyz*vec3(texelSize,1.0);
		vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize/RENDER_SCALE,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));
		gl_FragData[0] = texture
(texture, lmtexcoord.xy)*color;
		float avgBlockLum = luma(texture
Lod(texture, lmtexcoord.xy,128).rgb*color.rgb);
		gl_FragData[0].rgb = clamp((gl_FragData[0].rgb)*pow(avgBlockLum,-0.33)*0.85,0.0,1.0);
		vec3 albedo = toLinear(gl_FragData[0].rgb);
		
		if (iswater > 0.4) {
			albedo = vec3(0.42,0.6,0.7);
			gl_FragData[0] = vec4(0.42,0.6,0.7,0.7);
		}
		if (iswater > 0.9) {
			gl_FragData[0] = vec4(0.0);
		}


			vec3 normalTex = texture
(normals, lmtexcoord.xy).rgb;

			vec3 normal = normalMat.xyz;
			vec3 normal2 = normalMat.xyz;
			vec3 normal3 = normalMat.xyz;

			vec3 p3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;
			mat3 tbnMatrix = mat3(tangent.x, binormal.x, normal.x,
														tangent.y, binormal.y, normal.y,
														tangent.z, binormal.z, normal.z);
		//	if (iswater > 0.4){
				float bumpmult = 1.;
				float bumpmult2 = 0.1;
				float bumpmult3 = 20;
				if (iswater > 0.9)
					bumpmult = 1.;
				float parallaxMult = bumpmult;
				vec3 posxz = p3+cameraPosition;
				posxz.xz-=posxz.y;
				if (iswater < 0.9)
					posxz.xz *= 3.0;
				vec3 bump;
				vec3 bump2;
				vec3 bump3;


				posxz.xyz = getParallaxDisplacement(posxz,iswater,bumpmult,normalize(tbnMatrix*fragpos));
				bump = normalize(getWaveHeight(posxz.xz,iswater));
				bump2 = normalize(getWaveHeight(posxz.xz,iswater));
				bump3 = normalize(getWaveHeight(posxz.xz,iswater));
				if (iswater < 0.1) bump = normalTex;
				if (iswater < 0.1) bump2 = normalTex;
				if (iswater < 0.1) bump3 = normalTex;

				bump = bump * vec3(bumpmult, bumpmult, bumpmult) + vec3(0.0f, 0.0f, 1.0f - bumpmult);
				bump2 = bump2 * vec3(bumpmult2, bumpmult2, bumpmult2) + vec3(0.0f, 0.0f, 1.0f - bumpmult2);
				bump3 = bump3 * vec3(bumpmult3, bumpmult3, bumpmult3) + vec3(0.0f, 0.0f, 1.0f - bumpmult3);

				normal = normalize(bump * tbnMatrix);
				normal2 = normalize(bump2 * tbnMatrix);
				normal3 = normalize(bump3 * tbnMatrix);
		//	}

			float NdotL = lightSign*dot(normal,sunVec);
	
			float diffuseSun = clamp(NdotL,0.0f,1.0f);

			vec3 direct = texelFetch2D(gaux1,ivec2(6,37),0).rgb/3.1415;

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
					const float threshMul = max(2048.0/shadowMapResolution*shadowDistance*0.0078125,0.95);
					float distortThresh = (sqrt(1.0-diffuseSun*diffuseSun)/diffuseSun+0.7)/distortFactor;
					float diffthresh = distortThresh/6000.0*threshMul;

					projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5*0.166) + vec3(0.5,0.5,0.5);

					shading = 0.0;
					float noise = blueNoise();
					float rdMul = 4.0/shadowMapResolution;
					for(int i = 0; i < 9; i++){
						vec2 offsetS = tapLocation(i,9, 2.0,noise,0.0);

						float weight = 1.0+(i+noise)*rdMul/9.0*shadowMapResolution;
			#ifdef SHADOWS_ON			
						shading += shadow2D(shadow,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x*0.111;
			#else

					
			#endif			
						}
					direct *= shading;
				}
			}

			direct *= (iswater > 0.9 ? 0.2: 1.0)*diffuseSun*lmtexcoord.w;

			vec3 diffuseLight = direct + texture
(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize).rgb;
			
			vec3 color = diffuseLight*albedo*8./150./3.;


			if (iswater > 0.0){
			float f0 = iswater > 0.1?  0.02 : 0.05*(1.0-gl_FragData[0].a);

			float roughness = 0.02;



			vec3 reflectedVector = reflect(normalize(fragpos), normal);
			vec3 reflectedVector2 = reflect(normalize(fragpos), normal3);
			if (iswater < 0.1) reflectedVector = reflect(normalize(fragpos), normal2);
			float normalDotEye = dot(normal, normalize(fragpos));
			if (iswater < 0.1) normalDotEye = dot(normal, normalize(fragpos));
			float fresnel = Pow5(clamp(1.0 + normalDotEye,0.0,1.0));
				if (iswater < 0.1) fresnel = Pow5(clamp(0.5 + normalDotEye,0.0,1.0));
			fresnel = mix(f0,1.0,fresnel);
	
			if (iswater > 0.4){
				roughness = 0.1;
			}



			vec3 wrefl = mat3(gbufferModelViewInverse)*reflectedVector;
			if (iswater < 0.1) wrefl = mat3(gbufferModelViewInverse)*reflectedVector2;
			vec3 sky_c = mix(skyCloudsFromTex(wrefl,gaux1).rgb,texture
(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize).rgb*0.5,isEyeInWater);
			sky_c.rgb *= lmtexcoord.w*lmtexcoord.w*255*255/240./240./150.*8./3.;

			vec4 reflection = vec4(sky_c.rgb,0.);
			#ifdef SCREENSPACE_REFLECTIONS
			vec3 rtPos = rayTrace(reflectedVector,fragpos.xyz,blueNoise(), fresnel);
			if (rtPos.z <1.){
				vec3 previousPosition = mat3(gbufferModelViewInverse) * toScreenSpace(rtPos) + gbufferModelViewInverse[3].xyz + cameraPosition-previousCameraPosition;
				previousPosition = mat3(gbufferPreviousModelView) * previousPosition + gbufferPreviousModelView[3].xyz;
				previousPosition.xy = projMAD(gbufferPreviousProjection, previousPosition).xy / -previousPosition.z * 0.5 + 0.5;
				if (previousPosition.x > 0.0 && previousPosition.y > 0.0 && previousPosition.x < 1.0 && previousPosition.x < 1.0) {
					reflection.a = 1.0;
					reflection.rgb = texture
(gaux2,previousPosition.xy).rgb;
				}
			}
			#endif
			reflection.rgb = mix(sky_c.rgb, reflection.rgb, reflection.a);
			#ifdef SUN_MICROFACET_SPECULAR
				vec3 sunSpec = GGX(normal,-normalize(fragpos),  lightSign*sunVec, rainStrength*0.2+roughness+0.05+clamp(-lightSign*0.15,0.0,1.0), f0) * texelFetch2D(gaux1,ivec2(6,37),0).rgb*8./3./150.0/3.1415 * (1.0-rainStrength*0.9);
			#else
				vec3 sunSpec = drawSun(dot(lightSign*sunVec,reflectedVector), 0.0,texelFetch2D(gaux1,ivec2(6,37),0).rgb,vec3(0.0))*8./3./150.0*fresnel/3.1415 * (1.0-rainStrength*0.9);
				if (iswater < 0.1) sunSpec = drawSun(dot(lightSign*sunVec,reflectedVector2), 0.0,texelFetch2D(gaux1,ivec2(6,37),0).rgb,vec3(0.0))*8./3./150.0*fresnel/3.1415 * (1.0-rainStrength*0.9);
			#endif
			vec3 reflected= reflection.rgb*fresnel+shading*sunSpec;

		
			float alpha0 = gl_FragData[0].a;

			//correct alpha channel with fresnel
			gl_FragData[0].a = -gl_FragData[0].a*fresnel+gl_FragData[0].a+fresnel;
			gl_FragData[0].rgb =clamp(color/gl_FragData[0].a*alpha0*(1.0-fresnel)*0.1+reflected/gl_FragData[0].a*0.1,0.0,65100.0);
			if (gl_FragData[0].r > 65000.) gl_FragData[0].rgba = vec4(0.);
			}
			else
#ifndef nether
			gl_FragData[0].rgb = (color*(1+lumaboost))*0.1;
			
#endif


			gl_FragData[1] = vec4(0.1, 0.02, 0.0,iswater);
		    gl_FragData[2].rgba = vec4(normal3,1);

}
		}			

