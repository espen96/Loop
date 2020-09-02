#ifndef USE_LUMINANCE_AS_HEIGHTMAP
#ifndef MC_NORMAL_MAP
#undef PBR
#endif
#endif

#include "/lib/res_params.glsl"

#ifdef PBR
#define MC_NORMAL_MAP
#endif
varying vec4 vtexcoordam; // .st for add, .pq for mul
varying vec4 vtexcoord;
/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/

varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;

#ifndef water
varying vec4 tangent;
#else
varying vec3 tangent;
#endif
#ifdef water







varying vec3 binormal;

varying float dist;
varying vec3 viewVector;
#endif

varying vec2 texcoord;
attribute vec4 at_tangent;   

uniform float frameTimeCounter;
const float PI48 = 150.796447372*WAVY_SPEED;
float pi2wt = PI48*frameTimeCounter;
attribute vec4 mc_Entity;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
attribute vec4 mc_midTexCoord;
uniform vec3 cameraPosition;
uniform vec2 texelSize;
uniform int framemod8;
		const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
									vec2(-1.,3.)/8.,
									vec2(5.0,1.)/8.,
									vec2(-3,-5.)/8.,
									vec2(-5.,5.)/8.,
									vec2(-7.,-1.)/8.,
									vec2(3,7.)/8.,
									vec2(7.,-7.)/8.);
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec4 toClipSpace3(vec3 viewSpacePosition) {
    return vec4(projMAD(gl_ProjectionMatrix, viewSpacePosition),-viewSpacePosition.z);
}





uniform float rainStrength;
	
	vec3 calcWave(in vec3 pos, in float fm, in float mm, in float ma, in float f0, in float f1, in float f2, in float f3, in float f4, in float f5) {
		vec3 ret;


		float magnitude = sin(pi2wt*fm + pos.x*0.5 + pos.z*0.5 + pos.y*0.5) * mm + ma;
		

		float d0 = sin(pi2wt*f0);
		float d1 = sin(pi2wt*f1);
		float d2 = sin(pi2wt*f2);

		ret.x = sin(pi2wt*f3 + d0 + d1 - pos.x + pos.z + pos.y) * magnitude;
		ret.z = sin(pi2wt*f4 + d1 + d2 + pos.x - pos.z + pos.y) * magnitude;
		ret.y = sin(pi2wt*f5 + d2 + d0 + pos.z + pos.y - pos.y) * magnitude;

		return ret;
	}



float pow2(float x){return x*x;}
float pow15(float x){return pow2(pow2(pow2(x)*x)*x)*x;}


	vec3 doVertexDisplacement(vec3 viewpos, vec3 worldpos, vec4 lmcoord){
	
		float istopv = gl_MultiTexCoord0.t < mc_midTexCoord.t ? 1.0 : 0.0;
		
		float underCover = lmcoord.t;
			  underCover = clamp(pow15(underCover) * 2.0,0.0,1.0);
		

			float wavyMult = 1.0 + rainStrength*2;
			
			    vec3 move1 = calcWave(worldpos.xyz      , 0.003, 0.04, 0.04, 0.005, 0.009, 0.01, 0.006, 0.01, 0.01) * vec3(0.8,0.0,0.8);
				vec3 move2 = calcWave(worldpos.xyz+move1, 0.035, 0.04, 0.04, 0.004, 0.007, 0.005, 0.004, 0.006, 0.001) * vec3(0.8,0.0,0.48);


				float strength = 1 * WAVY_STRENGTH;

				move1 *= strength;
				move2 *= strength;
			
			
			
				vec3 waving =  move1+move2;
				     waving *= underCover * wavyMult;
			
			
				if ( mc_Entity.x == 10010 )
					{
						viewpos.xyz += waving * 0.1;
					}



				if ( mc_Entity.x == 10009 || mc_Entity.x == 10008  && istopv > 0.9 || mc_Entity.x == 10001 && istopv > 0.9|| mc_Entity.x == 10003  )
					{
						viewpos.xyz += waving;
					}

			return viewpos;
	}




//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
	lmtexcoord.xy = (gl_MultiTexCoord0).xy;
//	#ifdef PBR
	vec2 midcoord = (gl_TextureMatrix[0] *  mc_midTexCoord).st;
	vec2 texcoordminusmid = lmtexcoord.xy-midcoord;
	vtexcoordam.pq  = abs(texcoordminusmid)*2;
	vtexcoordam.st  = min(lmtexcoord.xy,midcoord-texcoordminusmid);
	vtexcoord.xy    = sign(texcoordminusmid)*0.5+0.5;
//	#endif
	
	
	vec2 lmcoord = gl_MultiTexCoord1.xy/255.;
	lmtexcoord.zw = lmcoord;

	vec3 position = mat3(gl_ModelViewMatrix) * vec3(gl_Vertex) + gl_ModelViewMatrix[3].xyz;

	texcoord = (gl_MultiTexCoord0).xy;							
	color = gl_Color;

	bool istopv = gl_MultiTexCoord0.t < mc_midTexCoord.t;
	#ifndef water
	#ifdef MC_NORMAL_MAP
		tangent = vec4(normalize(gl_NormalMatrix *at_tangent.rgb),at_tangent.w);
	#endif
	#endif
	

//	normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),0.5);	
//	normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),1.0);	
	
	
//    normalMat = vec4(normalMat.xy,sqrt(1.0 - dot(normalMat.xy, normalMat.xy)),0); 
	normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),mc_Entity.x == 10004 || mc_Entity.x == 10003 || mc_Entity.x == 10001 || mc_Entity.x == 10008 || mc_Entity.x == 10009 || mc_Entity.x == 10010 ? 0.0:1.0);
	
	

	
	
	
	#ifdef WAVY_PLANTS
	
	
	
	
	

    vec3 worldpos = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz + cameraPosition;
		
		
			vec4 lmcoord2 = gl_TextureMatrix[1] * gl_MultiTexCoord1;
			position = doVertexDisplacement(position, worldpos, lmcoord2);
		
		
	#endif
	
	
	
	
	
	
	if (mc_Entity.x == 10005){
		color.rgb = normalize(color.rgb)*sqrt(3.0);
		normalMat.a = 0.8;
	}
	if (mc_Entity.x == 10007){
		color.rgb = normalize(color.rgb)*sqrt(3.0);
		normalMat.a = 0.8;
	}
	
#ifdef hand	
 	normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),0.5);		
#endif	
//	gl_Position = ftransform();
	gl_Position = toClipSpace3(position);
	#ifdef SEPARATE_AO
	lmtexcoord.z *= sqrt(color.a);
	lmtexcoord.w *= color.a;
	#else
	
	color.rgb*=color.a;
	#endif
	#ifdef glint
		lmtexcoord.xy = (gl_TextureMatrix[0] * gl_MultiTexCoord0).st;
		lmtexcoord.zw = lmcoord*lmcoord;
	#endif
	
	
#ifdef water
//   position = mat3(gl_ModelViewMatrix) * vec3(gl_Vertex) + gl_ModelViewMatrix[3].xyz;
//  gl_Position = toClipSpace3(position.xyz);

	float mat = 0.0;
	if(mc_Entity.x == 8.0 || mc_Entity.x == 9.0) {
    mat = 1.0;
    gl_Position.z -= 1e-4;
  }


	if(mc_Entity.x == 79.0) mat = 0.5;
		if (mc_Entity.x == 10002) mat = 0.0001;
	normalMat = vec4(normalize( gl_NormalMatrix*gl_Normal),mat);

													  
					 
																											   
	   



	tangent = normalize( gl_NormalMatrix *at_tangent.rgb);
	binormal = normalize(cross(tangent.rgb,normalMat.xyz)*at_tangent.w);

	mat3 tbnMatrix = mat3(tangent.x, binormal.x, normalMat.x,
								  tangent.y, binormal.y, normalMat.y,
						     	  tangent.z, binormal.z, normalMat.z);

		dist = length(gl_ModelViewMatrix * gl_Vertex);
	viewVector = ( gl_ModelViewMatrix * gl_Vertex).xyz;
	viewVector = normalize(tbnMatrix * viewVector);	
	
#endif	
	
	#ifdef TAA_UPSCALING
		gl_Position.xy = gl_Position.xy * RENDER_SCALE + RENDER_SCALE * gl_Position.w - gl_Position.w;

	#endif
	#ifdef TAA
	gl_Position.xy += offsets[framemod8] * gl_Position.w*texelSize;
	#endif
	
}
