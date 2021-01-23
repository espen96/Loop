#version 120
#extension GL_EXT_gpu_shader4 : enable
#extension GL_ARB_shader_texture_lod : enable
#define SPEC


//#define POM
#define Depth_Write_POM	// POM adjusts the actual position, so screen space shadows can cast shadows on POM
#define POM_DEPTH 0.25 // [0.025 0.05 0.075 0.1 0.125 0.15 0.20 0.25 0.30 0.50 0.75 1.0] //Increase to increase POM strength
#define MAX_ITERATIONS 50 // [5 10 15 20 25 30 40 50 60 70 80 90 100 125 150 200 400] //Improves quality at grazing angles (reduces performance)
#define MAX_DIST 25.0 // [5.0 10.0 15.0 20.0 25.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 125.0 150.0 200.0 400.0] //Increases distance at which POM is calculated
//#define AutoGeneratePOMTextures	//Can generate POM on any texturepack (may look weird in some cases)
#define Texture_MipMap_Bias -1.00 // Uses a another mip level for textures. When reduced will increase texture detail but may induce a lot of shimmering. [-5.00 -4.75 -4.50 -4.25 -4.00 -3.75 -3.50 -3.25 -3.00 -2.75 -2.50 -2.25 -2.00 -1.75 -1.50 -1.25 -1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 3.75 4.00 4.25 4.50 4.75 5.00]
//#define DISABLE_ALPHA_MIPMAPS //Disables mipmaps on the transparency of alpha-tested things like foliage, may cost a few fps in some cases
#ifndef AutoGeneratePOMTextures
#ifndef MC_NORMAL_MAP
#undef POM
#endif
#endif

#ifdef POM
#define MC_NORMAL_MAP
#endif



#ifdef POM
const float mincoord = 1.0/4096.0;
const float maxcoord = 1.0-mincoord;

const float MAX_OCCLUSION_DISTANCE = MAX_DIST;
const float MIX_OCCLUSION_DISTANCE = MAX_DIST*0.9;
const int   MAX_OCCLUSION_POINTS   = MAX_ITERATIONS;
#endif



uniform vec2 texelSize;
#ifdef POM
varying vec4 vtexcoordam; // .st for add, .pq for mul
varying vec4 vtexcoord;

uniform int framemod8;
#endif
#include "/lib/res_params.glsl"
varying vec4 lmtexcoord;
varying vec4 color;
varying vec4 normalMat;
#ifdef MC_NORMAL_MAP
varying vec4 tangent;
uniform float wetness;
uniform sampler2D normals;
#endif

#ifdef SPEC
uniform sampler2D specular;
#endif

#ifdef POM
vec2 dcdx = dFdx(vtexcoord.st*vtexcoordam.pq)*exp2(Texture_MipMap_Bias);
vec2 dcdy = dFdy(vtexcoord.st*vtexcoordam.pq)*exp2(Texture_MipMap_Bias);
#endif
uniform sampler2D texture;
uniform float frameTimeCounter;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
float interleaved_gradientNoise(){
	return fract(52.9829189*fract(0.06711056*gl_FragCoord.x + 0.00583715*gl_FragCoord.y)+frameTimeCounter*51.9521);
}
#ifdef Depth_Write_POM
mat3 inverse(mat3 m) {
  float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
  float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
  float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

  float b01 = a22 * a11 - a12 * a21;
  float b11 = -a22 * a10 + a12 * a20;
  float b21 = a21 * a10 - a11 * a20;

  float det = a00 * b01 + a01 * b11 + a02 * b21;

  return mat3(b01, (-a22 * a01 + a02 * a21), (a12 * a01 - a02 * a11),
              b11, (a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10),
              b21, (-a21 * a00 + a01 * a20), (a11 * a00 - a01 * a10)) / det;
}
#endif
//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 n, vec2 lightmaps)
{

    return vec4(n.xy*inversesqrt(n.z*8.0+8.0) + 0.5,vec2(lightmaps.x,lightmaps.y));
}

#ifdef MC_NORMAL_MAP
vec3 applyBump(mat3 tbnMatrix, vec3 bump)
{

		float bumpmult = 1.0-wetness*0.95;

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
vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
#ifdef POM
vec4 readNormal(in vec2 coord)
{
	return texture2DGradARB(normals,fract(coord)*vtexcoordam.pq+vtexcoordam.st,dcdx,dcdy);
}
vec4 readTexture(in vec2 coord)
{
	return texture2DGradARB(texture,fract(coord)*vtexcoordam.pq+vtexcoordam.st,dcdx,dcdy);
}
#endif
float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}


vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}


const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
									vec2(-1.,3.)/8.,
									vec2(5.0,1.)/8.,
									vec2(-3,-5.)/8.,
									vec2(-5.,5.)/8.,
									vec2(-7.,-1.)/8.,
									vec2(3,7.)/8.,
									vec2(7.,-7.)/8.);
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* DRAWBUFFERS:17 */
void main() {
	float noise = interleaved_gradientNoise();
	vec3 normal = normalMat.xyz;
	#ifdef MC_NORMAL_MAP
		vec3 tangent2 = normalize(cross(tangent.rgb,normal)*tangent.w);
		mat3 tbnMatrix = mat3(tangent.x, tangent2.x, normal.x,
								  tangent.y, tangent2.y, normal.y,
						     	  tangent.z, tangent2.z, normal.z);
	#endif

#ifdef POM
vec2 tempOffset=offsets[framemod8];
vec2 adjustedTexCoord = fract(vtexcoord.st)*vtexcoordam.pq+vtexcoordam.st;
vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize/RENDER_SCALE,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));
vec3 viewVector = normalize(tbnMatrix*fragpos);
float dist = length(fragpos);
#ifdef Depth_Write_POM
gl_FragDepth = gl_FragCoord.z;
#endif
if (dist < MAX_OCCLUSION_DISTANCE) {
  #ifndef AutoGeneratePOMTextures
    if ( viewVector.z < 0.0 && readNormal(vtexcoord.st).a < 0.9999 && readNormal(vtexcoord.st).a > 0.00001) {
      vec3 interval = viewVector.xyz /-viewVector.z/MAX_OCCLUSION_POINTS*POM_DEPTH;
      vec3 coord = vec3(vtexcoord.st, 1.0);
      coord += noise*interval;
			float sumVec = noise;
      for (int loopCount = 0;
          (loopCount < MAX_OCCLUSION_POINTS) && (1.0 - POM_DEPTH + POM_DEPTH*readNormal(coord.st).a < coord.p) &&coord.p >= 0.0;
          ++loopCount) {
               coord = coord+interval;
							 sumVec += 1.0;
      }
      if (coord.t < mincoord) {
        if (readTexture(vec2(coord.s,mincoord)).a == 0.0) {
          coord.t = mincoord;
          discard;
        }
      }
      adjustedTexCoord = mix(fract(coord.st)*vtexcoordam.pq+vtexcoordam.st , adjustedTexCoord , max(dist-MIX_OCCLUSION_DISTANCE,0.0)/(MAX_OCCLUSION_DISTANCE-MIX_OCCLUSION_DISTANCE));


      #ifdef Depth_Write_POM
      vec3 truePos = fragpos + sumVec*inverse(tbnMatrix)*interval;	  
      gl_FragDepth = toClipSpace3(truePos).z;
      #endif
    }
  #else
	if ( viewVector.z < 0.0) {
		vec3 interval = viewVector.xyz/-viewVector.z/MAX_OCCLUSION_POINTS*POM_DEPTH;
		vec3 coord = vec3(vtexcoord.st, 1.0);
		coord += noise*interval;
		float sumVec = noise;
		float lum0 = luma(texture2DLod(texture,lmtexcoord.xy,100).rgb);
    for (int loopCount = 0;
        (loopCount < MAX_OCCLUSION_POINTS) && (1.0 - POM_DEPTH + POM_DEPTH*luma(readTexture(coord.st).rgb)/lum0*0.5 < coord.p) && coord.p >= 0.0;
				++loopCount) {
						 coord = coord+interval;
						 sumVec += 1.0;
		}
		if (coord.t < mincoord) {
			if (readTexture(vec2(coord.s,mincoord)).a == 0.0) {
				coord.t = mincoord;
				discard;
			}
		}
		adjustedTexCoord = mix(fract(coord.st)*vtexcoordam.pq+vtexcoordam.st , adjustedTexCoord , max(dist-MIX_OCCLUSION_DISTANCE,0.0)/(MAX_OCCLUSION_DISTANCE-MIX_OCCLUSION_DISTANCE));


		#ifdef Depth_Write_POM
		vec3 truePos = fragpos + sumVec*inverse(tbnMatrix)*(interval);		
		gl_FragDepth = toClipSpace3(truePos).z;
		#endif
	}
  #endif

  }

	vec4 data0 = texture2DGradARB(texture, adjustedTexCoord.xy,dcdx,dcdy);
  #ifdef DISABLE_ALPHA_MIPMAPS
    data0.a = texture2DGradARB(texture, adjustedTexCoord.xy,vec2(0.),vec2(0.0)).a;
  #endif
	if (data0.a > 0.1) data0.a = normalMat.a;
  else data0.a = 0.0;

	vec3 normalTex = texture2DGradARB(normals,adjustedTexCoord.xy,dcdx,dcdy).xyz;
	vec2 lm = lmtexcoord.zw*normalTex.b;
	normalTex.xy = normalTex.xy*2.0-1.0;

	
	normalTex.z = sqrt(1.0 - dot(normalTex.xy, normalTex.xy));
	normalTex.z = clamp(normalTex.z,0,1);	
	normal = applyBump(tbnMatrix,normalTex);


	data0.rgb*=color.rgb;
	vec4 data1 = clamp(noise*exp2(-8.)+encode(normal, lm),0.,1.0);

	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));

	#ifdef SPEC
	gl_FragData[1] = texture2DGradARB(specular, adjustedTexCoord.xy,dcdx,dcdy);
	gl_FragData[1].a = 0.0;
	#else 
	gl_FragData[1] = vec4(0.0);	
	#endif

	#else

	vec4 data0 = texture2D(texture, lmtexcoord.xy, Texture_MipMap_Bias);

	#ifdef SPEC
	gl_FragData[1] = texture2D(specular, lmtexcoord.xy, Texture_MipMap_Bias);
	#else
	gl_FragData[1] = vec4(0.0);
	#endif

	data0.rgb*=color.rgb;
  float avgBlockLum = luma(texture2DLod(texture, lmtexcoord.xy,128).rgb*color.rgb);
  data0.rgb = clamp(data0.rgb*pow(avgBlockLum,-0.33)*0.85,0.0,1.0);

  
  #ifdef DISABLE_ALPHA_MIPMAPS
  data0.a = texture2DLod(texture,lmtexcoord.xy,0).a;
  #endif


	if (data0.a > 0.1) data0.a = normalMat.a;
  else data0.a = 0.0;
	vec2 lm = lmtexcoord.zw;
	#ifdef MC_NORMAL_MAP
	vec3 normalTex = texture2D(normals, lmtexcoord.xy, Texture_MipMap_Bias).rgb;
	lm *= normalTex.b;
	normalTex.xy = normalTex.xy*2.0-1.0;

	normalTex.z = sqrt(1.0 - dot(normalTex.xy, normalTex.xy));
	normalTex.z = clamp(normalTex.z,0,1);			
	normal = applyBump(tbnMatrix,normalTex);

	#endif
	vec4 data1 = clamp(noise/256.+encode(normal, lm),0.,1.0);

	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));
	gl_FragData[1].a = 0.0;
	#endif

}
