#version 120
#extension GL_EXT_gpu_shader4 : enable
/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/

#include "/lib/Shadow_Params.glsl"
#include "/lib/settings.glsl"

#ifdef GI
#undef SHADOW_FRUSTRUM_CULLING
#endif

varying vec2 lightmaps;

const float PI = 3.1415927;
varying vec2 texcoord;

uniform mat4 shadowModelViewInverse;
uniform mat4 shadowModelView;

varying vec3 normal;							   
uniform vec3 cameraPosition;
varying vec4 lmcoord;

uniform float frameTimeCounter;

varying float iswater;

uniform float cosFov;
uniform vec3 shadowViewDir;
uniform vec3 shadowCamera;
uniform vec3 shadowLightVec;
uniform float shadowMaxProj;
attribute vec4 mc_Entity;
attribute vec4 mc_midTexCoord;
varying vec4 color;
const float PI48 = 150.796447372*WAVY_SPEED;
float pi2wt = PI48*frameTimeCounter;

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

	
vec2 calcWave(in vec3 pos) {

    float magnitude = abs(sin(dot(vec4(frameTimeCounter, pos),vec4(1.0,0.005,0.005,0.005)))*0.5+0.72)*0.013;
	vec2 ret = (sin(pi2wt*vec2(0.0063,0.0015)*4. - pos.xz + pos.y*0.05)+0.1)*magnitude;

    return ret;
}

vec3 calcMovePlants(in vec3 pos) {
    vec2 move1 = calcWave(pos );
	float move1y = -length(move1);
   return vec3(move1.x,move1y,move1.y)*5.*WAVY_STRENGTH/255.0;
}

vec3 calcWaveLeaves(in vec3 pos, in float fm, in float mm, in float ma, in float f0, in float f1, in float f2, in float f3, in float f4, in float f5) {

    float magnitude = abs(sin(dot(vec4(frameTimeCounter, pos),vec4(1.0,0.005,0.005,0.005)))*0.5+0.72)*0.013;
	vec3 ret = (sin(pi2wt*vec3(0.0063,0.0224,0.0015)*1.5 - pos))*magnitude;

    return ret;
}

vec3 calcMoveLeaves(in vec3 pos, in float f0, in float f1, in float f2, in float f3, in float f4, in float f5, in vec3 amp1, in vec3 amp2) {
    vec3 move1 = calcWaveLeaves(pos      , 0.0054, 0.0400, 0.0400, 0.0127, 0.0089, 0.0114, 0.0063, 0.0224, 0.0015) * amp1;
    return move1*5.*WAVY_STRENGTH/255.;
}	




bool intersectCone(float coneHalfAngle, vec3 coneTip , vec3 coneAxis, vec3 rayOrig, vec3 rayDir, float maxZ)
{
  vec3 co = rayOrig - coneTip;
  float prod = dot(normalize(co),coneAxis);
  if (prod <= -coneHalfAngle) return true;   //In view frustrum

  float a = dot(rayDir,coneAxis)*dot(rayDir,coneAxis) - coneHalfAngle*coneHalfAngle;
  float b = 2.1 * (dot(rayDir,coneAxis)*dot(co,coneAxis) - dot(rayDir,co)*coneHalfAngle*coneHalfAngle);
  float c = dot(co,coneAxis)*dot(co,coneAxis) - dot(co,co)*coneHalfAngle*coneHalfAngle;

   float det = b*b - 4.*a*c; 
   
   

  if (det < 0.) return false;    // No intersection with either forward cone and backward cone
  det = sqrt(det);
  float t2 = (-b + det) / (2. * a);
  if (t2 <= 0.0 || t2 >= maxZ) return false;  //Idk why it works
  return true;
}



#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec4 toClipSpace3(vec3 viewSpacePosition) {
    return vec4(projMAD(gl_ProjectionMatrix, viewSpacePosition),1.0);
}
void main() {
	color = gl_Color;

	vec3 position = mat3(gl_ModelViewMatrix) * vec3(gl_Vertex) + gl_ModelViewMatrix[3].xyz;
  //Check if the vertice is going to cast shadows
  #ifdef SHADOW_FRUSTRUM_CULLING
  if (intersectCone(cosFov, shadowCamera, shadowViewDir, position, -shadowLightVec, shadowMaxProj)) {
  #endif


  	#ifdef WAVY_PLANTS
	
	
	
	
	

      vec3 worldpos = mat3(shadowModelViewInverse) * position + shadowModelViewInverse[3].xyz;
		
		
			vec4 lmcoord2 = gl_TextureMatrix[1] * gl_MultiTexCoord1;
			position = doVertexDisplacement(position, worldpos, lmcoord2);
		
		
	#endif
  
		iswater = 0.0;
		lightmaps = gl_MultiTexCoord1.xy * (1.0 / 255.0);
	if (mc_Entity.x == 8.0)
	iswater = 1.0;
  
	gl_Position = BiasShadowProjection(toClipSpace3(position));
	gl_Position.z /= 6.0;
	lmcoord = gl_TextureMatrix[1] * gl_MultiTexCoord1;
	texcoord.xy = gl_MultiTexCoord0.xy;
	normal = normalize(gl_NormalMatrix * gl_Normal);		 
  #ifdef SHADOW_FRUSTRUM_CULLING
  }
  else
  gl_Position.xyzw = vec4(0.0,0.0,1e30,0.0);  //Degenerates the triangle
  #endif
}
