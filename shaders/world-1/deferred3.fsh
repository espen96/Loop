#version 150 compatibility
#extension GL_EXT_gpu_shader4 : enable

#define TAA
#define TORCH_R 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_G 0.5 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define TORCH_B 0.2 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]

flat in vec3 WsunVec;
flat in vec2 TAA_Offset;
#include "/lib/res_params.glsl"
#include "/lib/Shadow_Params.glsl"
uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D colortex1;
uniform sampler2D colortex5;
uniform sampler2D colortex6;
uniform sampler2D colortex10;
uniform sampler2D colortex13;
uniform sampler2D colortex15;
uniform sampler2D colortex2;

uniform sampler2D shadow;
uniform sampler2D noisetex;
uniform vec3 sunVec;
uniform vec2 viewSize;
uniform vec2 texelSize;
uniform float frameTimeCounter;
uniform float rainStrength;
uniform int frameCounter;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
in vec2 coord;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform vec3 cameraPosition;
uniform float viewWidth;
uniform float aspectRatio;
uniform float viewHeight;
uniform float far;
uniform float near;

#include "/lib/color_transforms.glsl"
#include "/lib/noise.glsl"
#include "/lib/encode.glsl"
#include "/lib/kernel.glsl"

#define ffstep(x,y) clamp((y - x) * 1e35,0.0,1.0)
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter,float distort)
{
    float alpha = (sampleNumber+jitter)/nb;
    float angle = jitter*6.28+alpha * nbRot * 6.28;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*alpha;
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


vec2 moment(ivec2 pos) {

float weightSum = 1.0;
vec2 moment = texelFetch(colortex15, pos, 0).rg;
float depth = texelFetch(depthtex0, pos, 0).x;
vec3 normal = texelFetch(colortex10, pos, 0).xyz;
    for (int i = 0; i<9; i++) {
        ivec2 deltaPos     = kernelO_3x3[i]*8;
        //  We already have the center data
//        if (pos != 0 && pos != 0) { continue; }

        // ⬇️ Sample current point data with current uv
        ivec2 p = pos + deltaPos;
        vec4 curColor = texelFetch(colortex5, ivec2(p/RENDER_SCALE), 0);
        float curDepth = texelFetch(depthtex0, p, 0).x;
        vec3 curNormal = texelFetch(colortex10, p, 0).xyz;

        //  Determine the average brightness of this sample
        //  Using International Telecommunications Union's ITU BT.601 encoding params
        float l = luma(curColor.rgb);

        float weightDepth = abs(curDepth - depth) / (depth * length(vec2(deltaPos)) + 1.0e-2);
        float weightNormal = pow(max(0, dot(curNormal, normal)), 32.0);
        float w = exp(-weightDepth) * weightNormal;


        weightSum += w;

        moment += vec2(l) * w;
    }


moment /= weightSum;

    return  moment;
}


void main() {
/* RENDERTARGETS: 3 */
	vec2 texcoord = ((gl_FragCoord.xy))*texelSize;
//	gl_FragData[1].rg =  moment(ivec2(floor(texcoord * vec2(viewWidth, viewHeight)))) ;
	gl_FragData[0] = vec4(Min_Shadow_Filter_Radius, 0.1, 0.0, 0.0);
	float z = texture(depthtex1,texcoord).x;
	vec2 tempOffset=TAA_Offset;
	if (z < 1.0){

		vec4 data = texture(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		vec3 normal = mat3(gbufferModelViewInverse) * worldToView(decode(dataUnpacked0.yw));
		vec2 lightmap = dataUnpacked1.yz;
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;
		bool translucent2 = abs(dataUnpacked1.w-0.6) <0.01;	// Weak translucency
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));



		if (!hand){
			float NdotL = clamp(dot(normal,WsunVec),0.0,1.0);
			float bn = blueNoiseFloat(gl_FragCoord.xy);
			float noise = fract(bn + frameCounter/1.6180339887);
			vec3 fragpos = toScreenSpace(vec3(texcoord/RENDER_SCALE-vec2(tempOffset)*texelSize*0.5,z));
		#ifdef Variable_Penumbra_Shadows
				if (NdotL > 0.001 || translucent || translucent2) {
				vec3 p3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;
				vec3 projectedShadowPosition = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
				projectedShadowPosition = diagonal3(shadowProjection) * projectedShadowPosition + shadowProjection[3].xyz;

				//apply distortion
				float distortFactor = calcDistort(projectedShadowPosition.xy);
				projectedShadowPosition.xy *= distortFactor;
				//do shadows only if on shadow map
				if (abs(projectedShadowPosition.x) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.y) < 1.0-1.5/shadowMapResolution && abs(projectedShadowPosition.z) < 6.0){
					const float threshMul = max(2048.0/shadowMapResolution*shadowDistance/128.0,0.95);
					float distortThresh = (sqrt(1.0-NdotL*NdotL)/NdotL+0.7)/distortFactor;
					float diffthresh = distortThresh/6000.0*threshMul;
					projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);

					const float mult = Max_Shadow_Filter_Radius;
					float avgBlockerDepth = 0.0;
					vec2 scales = vec2(0.0,Max_Filter_Depth);
					float blockerCount = 0.0;
					float rdMul = distortFactor*(1.0+mult)*d0*k/shadowMapResolution;
					float diffthreshM = diffthresh*mult*d0*k/20.;
					float avgDepth = 0.0;
					for(int i = 0; i < VPS_Search_Samples; i++){
						vec2 offsetS = tapLocation(i,VPS_Search_Samples, 84.0, noise,0.0);
						float weight = 3.0 + (i+noise) *rdMul/SHADOW_FILTER_SAMPLE_COUNT*shadowMapResolution*distortFactor/2.7;
						float d = texelFetch2D( shadow, ivec2((projectedShadowPosition.xy+offsetS*rdMul)*shadowMapResolution),0).x;
						float b = smoothstep(weight*diffthresh/2.0, weight*diffthresh, projectedShadowPosition.z - d);

						blockerCount += b;
						avgDepth += max(projectedShadowPosition.z - d, 0.0)*1000.;
						avgBlockerDepth += d * b;
					}
					gl_FragData[0].g = avgDepth / VPS_Search_Samples;
					gl_FragData[0].b = blockerCount / VPS_Search_Samples;
					if (blockerCount >= 0.9){
						avgBlockerDepth /= blockerCount;
						float ssample = max(projectedShadowPosition.z - avgBlockerDepth,0.0)*1500.0;
						gl_FragData[0].r = clamp(ssample, scales.x, scales.y)/(scales.y)*(mult-Min_Shadow_Filter_Radius)+Min_Shadow_Filter_Radius;
					}
				}
			}
		#endif
		}

}
}
