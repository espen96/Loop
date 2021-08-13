

/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/
varying vec3 velocity;
//attribute vec3 at_velocity;   


varying vec2 texcoord;
varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;
varying float mcentity;
attribute vec4 mc_Entity;
attribute vec4 mc_midTexCoord;
uniform int entityId; 
#ifdef water
varying vec3 binormal;
varying vec3 tangent;					 
varying float dist;
varying float lumaboost;
varying vec3 viewVector;


attribute vec4 at_tangent;
uniform mat4 gbufferModelViewInverse;
#else

#ifdef MC_NORMAL_MAP
varying vec4 tangent;
attribute vec4 at_tangent;
#endif
#endif
uniform int blockEntityId;

#ifdef solid1
uniform vec3 cameraPosition;

uniform mat4 gbufferModelViewInverse;
uniform float frameTimeCounter;
#endif


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

//#define POM							
#if defined (POM)||  defined (DLM)
varying vec4 vtexcoordam; // .st for add, .pq for mul
varying vec4 vtexcoord;
#endif					   
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

#ifdef solid1								

#endif		
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec4 toClipSpace3(vec3 viewSpacePosition) {
    return vec4(projMAD(gl_ProjectionMatrix, viewSpacePosition),-viewSpacePosition.z);
}														
								   

																																		
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {

											
	vec2 lmcoord = gl_MultiTexCoord1.xy/255.;
  #if defined (POM)||  defined (DLM)
	vec2 midcoord = (gl_TextureMatrix[0] *  mc_midTexCoord).st;
	vec2 texcoordminusmid = lmtexcoord.xy-midcoord;
	vtexcoordam.pq  = abs(texcoordminusmid)*2;
	vtexcoordam.st  = min(lmtexcoord.xy,midcoord-texcoordminusmid);
	vtexcoord.xy    = sign(texcoordminusmid)*0.5+0.5;
	#endif 
	
#if defined(weather) || defined(glint)	
		lmtexcoord.zw = lmcoord*lmcoord;	
	#else
		lmtexcoord.zw = lmcoord;
#endif


#ifdef glint 
		lmtexcoord.xy = (gl_TextureMatrix[0] * gl_MultiTexCoord0).st;	
	#else	
		lmtexcoord.xy = (gl_MultiTexCoord0).xy;	
#endif


	texcoord = (gl_MultiTexCoord0).xy;
	
#if defined(solid1) || defined(water) || defined(hand)			
	vec3 position = mat3(gl_ModelViewMatrix) * vec3(gl_Vertex) + gl_ModelViewMatrix[3].xyz;	
#else
	gl_Position = toClipSpace3(mat3(gl_ModelViewMatrix) * vec3(gl_Vertex) + gl_ModelViewMatrix[3].xyz);
#endif	
	
#ifdef solid1	
	
	vec3 worldpos = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz + cameraPosition;
	bool istopv = worldpos.y > cameraPosition.y+5.0;
	float ft = frameTimeCounter*1.3;
	if (!istopv) position.xz += vec2(3.0,1.0)+sin(ft)*sin(ft)*sin(ft)*vec2(2.1,0.6);
	position.xz -= (vec2(3.0,1.0)+sin(ft)*sin(ft)*sin(ft)*vec2(2.1,0.6))*0.5;
	gl_Position = toClipSpace3(position);	
#endif

#if defined(solid1) || defined(water) || defined(hand)		
  gl_Position = toClipSpace3(position);	
#endif  
	

	color = gl_Color;

#ifdef water
	lumaboost = 0.0;
	float mat = 0.0;
	if(mc_Entity.x == 20) {
    mat = 1.0;
    gl_Position.z -= 1e-4;
  }
		if (mc_Entity.x == 11) lumaboost = 5.0;

	if(mc_Entity.x == 79.0) mat = 0.5;
		if (mc_Entity.x == 10002) mat = 0.01;
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
	
	
	#ifndef water
	#ifdef MC_NORMAL_MAP
		tangent = vec4(normalize(gl_NormalMatrix *at_tangent.rgb),at_tangent.w);
	#endif
	#endif
	#ifdef normal1
		normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),blockEntityId==10006? 1.0:1.0);	
		if (entityId == 18)	normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),1.0);
	#else
	#ifndef water
	#ifdef hand
	
	normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),0.5);
	#else
		normalMat = vec4(normalize(gl_NormalMatrix *gl_Normal),0.0);
	#endif
	#endif
	#endif
	
//	velocity = at_velocity / 1000000.0;
	velocity = vec3(0.0);

	#ifdef TAA_UPSCALING
		gl_Position.xy = gl_Position.xy * RENDER_SCALE + RENDER_SCALE * gl_Position.w - gl_Position.w;
	#endif
	#ifdef TAA
	gl_Position.xy += offsets[framemod8] * gl_Position.w*texelSize;
	#endif
	mcentity = 0;
	if(mc_Entity.x >16) mcentity = mc_Entity.x;
}
