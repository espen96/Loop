#version 150
#extension GL_EXT_gpu_shader4 : enable
/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/
//#define SHADOW_FRUSTRUM_CULLING   // If enabled, removes most of the blocks during shadow rendering that would not cast shadows on the player field of view. Improves performance but can be sometimes incorrect and causes flickering shadows on distant occluders
#define WAVY_PLANTS
#define WAVY_STRENGTH 1.0 //[0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0]
#define WAVY_SPEED 1.0 //[0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.25 1.5 2.0 3.0 4.0]
#include "/lib/Shadow_Params.glsl"

// Compatibility
#extension GL_EXT_gpu_shader4 : enable
in vec3 vaPosition;
in vec4 vaColor;
in vec2 vaUV0;
in ivec2 vaUV2;
in vec3 vaNormal;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 textureMatrix = mat4(1.0);
uniform mat3 normalMatrix;
uniform vec3 chunkOffset;
out vec2 texcoord;

uniform mat4 shadowModelViewInverse;
uniform mat4 shadowModelView;
uniform vec3 cameraPosition;
uniform float frameTimeCounter;
uniform float cosFov;
uniform vec3 shadowViewDir;
uniform vec3 shadowCamera;
uniform vec3 shadowLightVec;
uniform float shadowMaxProj;
in vec4 mc_Entity;
in vec4 mc_midTexCoord;

const float PI48 = 150.796447372*WAVY_SPEED;
float pi2wt = PI48*frameTimeCounter;

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
  float b = 2. * (dot(rayDir,coneAxis)*dot(co,coneAxis) - dot(rayDir,co)*coneHalfAngle*coneHalfAngle);
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
    return vec4(projMAD(projectionMatrix, viewSpacePosition),1.0);
}
void main() {

	vec3 position = mat3(modelViewMatrix) * vec3(vec4(vaPosition + chunkOffset, 1.0)) + modelViewMatrix[3].xyz;
  //Check if the vertice is going to cast shadows
  #ifdef SHADOW_FRUSTRUM_CULLING
  if (intersectCone(cosFov, shadowCamera, shadowViewDir, position, -shadowLightVec, shadowMaxProj)) {
  #endif
	#ifdef WAVY_PLANTS
  	bool istopv = vaUV0.t < mc_midTexCoord.t;
    if ((mc_Entity.x == 10001&&istopv) && length(position.xy) < 24.0) {
      vec3 worldpos = mat3(shadowModelViewInverse) * position + shadowModelViewInverse[3].xyz;
      worldpos.xyz += calcMovePlants(worldpos.xyz + cameraPosition)*vec4(vaUV2, 0.0, 1.0).y;
      position = mat3(shadowModelView) * worldpos + shadowModelView[3].xyz ;
    }

    if ((mc_Entity.x == 10003) && length(position.xy) < 24.0) {
      vec3 worldpos = mat3(shadowModelViewInverse) * position + shadowModelViewInverse[3].xyz;
      worldpos.xyz += calcMoveLeaves(worldpos.xyz + cameraPosition, 0.0040, 0.0064, 0.0043, 0.0035, 0.0037, 0.0041, vec3(1.0,0.2,1.0), vec3(0.5,0.1,0.5))*vec4(vaUV2, 0.0, 1.0).y;
      position = mat3(shadowModelView) * worldpos + shadowModelView[3].xyz ;
    }
  #endif
	gl_Position = BiasShadowProjection(toClipSpace3(position));
	gl_Position.z /= 6.0;

	texcoord.xy = vaUV0.xy;
	if(mc_Entity.x == 20) gl_Position.w = -1.0;
  #ifdef SHADOW_FRUSTRUM_CULLING
  }
  else
  gl_Position.xyzw = vec4(0.0,0.0,1e30,0.0);  //Degenerates the triangle
  #endif
}
