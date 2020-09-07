#version 120
//Render sky, volumetric clouds, direct lighting
#extension GL_EXT_gpu_shader4 : enable
#include "/lib/settings.glsl"

varying vec2 texcoord;

flat varying vec2 TAA_Offset;
flat varying float tempOffsets;


uniform sampler2D colortex1;
uniform sampler2D colortex2;
uniform sampler2D colortex3;
uniform sampler2D colortex7;
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth



uniform sampler2D shadow;
uniform sampler2D shadowcolor1;
uniform sampler2D shadowcolor0;
uniform sampler2D shadowtex1;



uniform int frameCounter;
uniform float frameTime;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 previousCameraPosition;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;
uniform mat4 gbufferModelView;
uniform int worldTime;


uniform vec2 texelSize;

uniform vec3 cameraPosition;



vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}

#include "/lib/res_params.glsl"

#include "/lib/Shadow_Params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/util.glsl"
#include "/lib/encode.glsl"



vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}


vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}




#include "/lib/rsm.glsl"


void main() {




vec3 shadowCol = vec3(0.0);


#ifdef GI
	vec2 texcoord = gl_FragCoord.xy*texelSize/RSM_SCALE;							 
	float z = texture2D(depthtex1,texcoord).x;
	if (z <=1.0) {
		float masks = texture2D(colortex3,texcoord).a;
		bool iswater = texture2D(colortex3,texcoord).a > 0.9;

		vec4 data = texture2D(colortex1,texcoord);

		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 normal = mat3(gbufferModelViewInverse) * decode(dataUnpacked0.yw);
		vec3 normal2 = decode(dataUnpacked0.yw);
		vec2 lightmap = vec2(dataUnpacked1.yz);			

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));		
		
		bool translucent = abs(dataUnpacked1.w-0.5) <0.01;
		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;
		bool glass = texture2D(colortex2,texcoord).a >=0.01;			
		bool hand = abs(dataUnpacked1.w-0.75) <0.01;
		bool entity = (masks) <=0.10 && (masks) >=0.09;

	

		
				if(!entity && !iswater && !translucent && !emissive && !glass){
				
					shadowCol = (getRSM(normal,entity,albedo, lightmap,z)*vec3(1.2,1.2,1.2)*5) * rsmStrength;
					float lum = luma(shadowCol);
					vec3 diff = shadowCol-lum;		


  
					#define GISAT 1.0
					#define GICROSS -5.0
				
					shadowCol = shadowCol + diff*(-lum*(GICROSS) + GISAT);



			
				}
			


	
			
		
		
		}
	
	
		#endif		
		gl_FragData[0].rgb  = shadowCol;		
	

	
	
/* DRAWBUFFERS:7 */
}
