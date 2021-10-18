#define gbuffer
     uniform int renderStage; 
in float mcentity;
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

uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform vec3 previousCameraPosition;
uniform vec3 cameraPosition;
in vec3 velocity;
#define DLM
in float nonlabemissive;
//#define POM
#define labspec
#define bumpmultiplier 1.0
//#define smooth_depth


#define Depth_Write_POM	// POM adjusts the actual position, so screen space shadows can cast shadows on POM
#define POM_DEPTH 0.25 // [0.025 0.05 0.075 0.1 0.125 0.15 0.20 0.25 0.30 0.50 0.75 1.0] //Increase to increase POM strength
#define MAX_ITERATIONS 50 // [5 10 15 20 25 30 40 50 60 70 80 90 100 125 150 200 400] //Improves quality at grazing angles (reduces performance)
#define MAX_DIST 25.0 // [5.0 10.0 15.0 20.0 25.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0 125.0 150.0 200.0 400.0] //Increases distance at which POM is calculated
//#define AutoGeneratePOMTextures	//Can generate POM on any texturepack (may look weird in some cases)
#define Texture_MipMap_Bias 0.00 // Uses a another mip level for textures. When reduced will increase texture detail but may induce a lot of shimmering. [-5.00 -4.75 -4.50 -4.25 -4.00 -3.75 -3.50 -3.25 -3.00 -2.75 -2.50 -2.25 -2.00 -1.75 -1.50 -1.25 -1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00 3.25 3.50 3.75 4.00 4.25 4.50 4.75 5.00]
//#define DISABLE_ALPHA_MIPMAPS //Disables mipmaps on the transparency of alpha-tested things like foliage, may cost a few fps in some cases
#ifndef AutoGeneratePOMTextures


#ifndef MC_NORMAL_MAP
#undef POM
#endif
#endif



#ifdef SPEC
uniform sampler2D specular;
#endif

#if defined (POM)||  defined (DLM)

in vec4 vertexPos;
#define MC_NORMAL_MAP

const float mincoord = 1.0/4096.0;
const float maxcoord = 1.0-mincoord;

const float MAX_OCCLUSION_DISTANCE = MAX_DIST;
const float MIX_OCCLUSION_DISTANCE = MAX_DIST*0.9;
const int   MAX_OCCLUSION_POINTS   = MAX_ITERATIONS;

in vec4 vtexcoordam; // .st for add, .pq for mul
in vec4 vtexcoord;

uniform int framemod8;

vec2 dcdx = dFdx(vtexcoord.st*vtexcoordam.pq)*exp2(Texture_MipMap_Bias);
vec2 dcdy = dFdy(vtexcoord.st*vtexcoordam.pq)*exp2(Texture_MipMap_Bias);
#endif

in vec2 taajitter;

#ifdef MC_NORMAL_MAP
in vec4 tangent;
uniform float wetness;
uniform sampler2D normals;
#endif
uniform sampler2D noisetex;//depth
uniform float far;
in vec4 lmtexcoord;
in vec4 color;
in vec4 normalMat;
uniform vec2 texelSize;
uniform sampler2D texture;
uniform sampler2D gaux1;
uniform sampler2D depthtex1;
uniform float frameTimeCounter;
uniform int frameCounter;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;
in vec4 hspec;

uniform ivec2 atlasSize;  


uniform float viewWidth;
uniform float viewHeight;

#include "/lib/noise.glsl"


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

vec3 toWorldSpace(vec3 p3){
    p3 = mat3(gbufferModelViewInverse) * p3 + gbufferModelViewInverse[3].xyz;
    return p3;
}


//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 unenc, vec2 lightmaps)
{
	unenc.xy = unenc.xy / dot(abs(unenc), vec3(1.0)) + 0.00390625;
	unenc.xy = unenc.z <= 0.0 ? (1.0 - abs(unenc.yx)) * sign(unenc.xy) : unenc.xy;
    vec2 encn = unenc.xy * 0.5 + 0.5;
	
    return vec4((encn),vec2(lightmaps.x,lightmaps.y));
}

#ifdef MC_NORMAL_MAP
vec3 applyBump(mat3 tbnMatrix, vec3 bump)
{

		float bumpmult = (1.0-wetness*0.95)*bumpmultiplier;

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


vec3 toScreenSpace(vec3 p) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x, gbufferProjectionInverse[1].y, gbufferProjectionInverse[2].zw);
    vec3 p3 = p * 2. - 1.;
    vec4 fragposition = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return fragposition.xyz / fragposition.w;
}
#ifdef POM


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

#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define  projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)

vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}







float smoothDepth(vec2 coord){

ivec2 tileResolution = ivec2(atlasSize*vtexcoordam.pq+0.5);
ivec2 tileOffset     = ivec2(atlasSize*vtexcoordam.st+0.5);
				coord = coord * atlasSize*vtexcoordam.pq;
				ivec2 i = ivec2(coord);
				vec2  f = fract(coord);

				float s0 = texture2DGradARB(normals, (mod((i + ivec2(0, 1)), tileResolution) + tileOffset)/vec2(atlasSize), dcdx,dcdy).a;
				float s1 = texture2DGradARB(normals, (mod((i + ivec2(1, 1)), tileResolution) + tileOffset)/vec2(atlasSize), dcdx,dcdy).a;
				float s2 = texture2DGradARB(normals, (mod((i + ivec2(1, 0)), tileResolution) + tileOffset)/vec2(atlasSize), dcdx,dcdy).a;
				float s3 = texture2DGradARB(normals, (mod((i + ivec2(0, 0)), tileResolution) + tileOffset)/vec2(atlasSize), dcdx,dcdy).a;

				return mix(mix(s3, s2, f.x), mix(s0, s1, f.x), f.y);
			}


vec4 readNormal(in vec2 coord)
{

#ifdef smooth_depth
	return vec4(smoothDepth(coord));
#else
	return texture2DGradARB(normals,fract(coord)*vtexcoordam.pq+vtexcoordam.st,dcdx,dcdy);
#endif	
}
vec4 readTexture(in vec2 coord)
{
	return texture2DGradARB(texture,fract(coord)*vtexcoordam.pq+vtexcoordam.st,dcdx,dcdy);
}
#endif
float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}



const vec2[8] offsets = vec2[8](vec2(1./8.,-3./8.),
									vec2(-1.,3.)/8.,
									vec2(5.0,1.)/8.,
									vec2(-3,-5.)/8.,
									vec2(-5.,5.)/8.,
									vec2(-7.,-1.)/8.,
									vec2(3,7.)/8.,
									vec2(7.,-7.)/8.);
	vec3 ToNDC(vec3 pos) {
	vec4 iProjDiag = vec4(gbufferProjectionInverse[0].x,
						  gbufferProjectionInverse[1].y,
						  gbufferProjectionInverse[2].zw);
    vec3 p3 = pos * 2.0 - 1.0;
    vec4 viewPos = iProjDiag * p3.xyzz + gbufferProjectionInverse[3];
    return viewPos.xyz / viewPos.w;
}





#ifdef POM

       vec3 FindNormal(sampler2D tex, vec2 uv, vec2 u)
            {
                    //u is one uint size, ie 1.0/texture size
                vec2 offsets[4];
					 offsets[0] = uv + vec2(-u.x, 0);
					 offsets[1] = uv + vec2(u.x, 0);
					 offsets[2] = uv + vec2(0, -u.x);
					 offsets[3] = uv + vec2(0, u.x);
               
                float hts[4];
                for(int i = 0; i < 4; i++)
                {
				

                    hts[i] = length(texture(tex, offsets[i]).x); 			

                }
               
                vec2 _step = vec2(0.1, 0.0);
               
			   
                vec3 va = normalize( vec3(_step.xy, hts[1]-hts[0]) );
                vec3 vb = normalize( vec3(_step.yx, hts[3]-hts[2]) );
				
	            if (vtexcoord.x > 1.0 - 0.01 || vtexcoord.y > 1.0 - 0.01)  return vec3(0.0);   
	            if (vtexcoord.x < 0.01 || vtexcoord.y < 0.01)              return vec3(0.0);
			   
                return cross(va,vb).rgb; //you may not need to swizzle the normal

               
            }
#endif 








float lmfaloff(float lm)
{
	float falloff = 10.0;

	lm = exp(-(1.0 - lm) * falloff);
	lm = max(0.0, lm - exp(-falloff));

	return lm;
}


#define viewMAD(m, v) (mat3(m) * (v) + (m)[3].xyz)

vec3 reproject(vec3 sceneSpace) {
    vec3 prevScreenPos = cameraPosition - previousCameraPosition;

    prevScreenPos = prevScreenPos*0.5;

  
    return prevScreenPos;
}  
mat3 getLightmapTBN(vec3 viewPos){
    mat3 lmTBN = mat3(normalize(dFdx(viewPos)), normalize(dFdy(viewPos)), vec3(0.0));
    lmTBN[2] = cross(lmTBN[0], lmTBN[1]);
    return lmTBN;
}
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
/* RENDERTARGETS: 1,7,10,11 */
void main() {





		vec3 screenPos = vec3(gl_FragCoord.xy / vec2(viewWidth, viewHeight), gl_FragCoord.z);
		vec3 viewPos = ToNDC(screenPos);
		float dither = blueNoise();
		float noise = dither;
//			  dither = fract(frameTimeCounter * 16.0 + dither);
//		vec3 worldPos = toWorldSpace(viewPos);				
//		float fogFactorAbs = 1.0 - clamp(((length(worldPos)-2) - gl_Fog.start) * gl_Fog.scale, 0.0, 1.0);
//		if (fogFactorAbs < dither) discard;
		
		
	vec3 albedo = textureLod(texture, lmtexcoord.xy, Texture_MipMap_Bias).xyz;

	vec3 normal = normalMat.xyz;
	#ifdef MC_NORMAL_MAP
		vec3 tangent2 = normalize(cross(tangent.rgb,normal)*tangent.w);
		mat3 tbnMatrix = mat3(tangent.x, tangent2.x, normal.x,
								  tangent.y, tangent2.y, normal.y,
						     	  tangent.z, tangent2.z, normal.z);
				
	#endif

vec2 lm = lmtexcoord.zw;
	


//////////////////////////////POM//////////////////////////////	

	vec2 tempOffset=offsets[framemod8];	
		vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize/RENDER_SCALE,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));	
	

#ifdef POM


	
		vec2 adjustedTexCoord = fract(vtexcoord.st)*vtexcoordam.pq+vtexcoordam.st;



	


		  vec3 coord = vec3(vtexcoord.st, 1.0);
		  
		vec3 viewVector = normalize(tbnMatrix*fragpos);
		float dist = length(fragpos);
		#ifdef Depth_Write_POM
		gl_FragDepth = gl_FragCoord.z;
		#endif
		if (dist < MAX_OCCLUSION_DISTANCE) {
		  #ifndef AutoGeneratePOMTextures
			if ( viewVector.z < 0.0 && readNormal(vtexcoord.st).a < 0.9999 && readNormal(vtexcoord.st).a > 0.00001) {
			  vec3 interval = viewVector.xyz /-viewVector.z/MAX_OCCLUSION_POINTS*POM_DEPTH;
			  vec2 atlasAspect = vec2(atlasSize.y/float(atlasSize.x),atlasSize.x/float(atlasSize.y));
				vec2 viewCorrection = max(vec2((vtexcoordam.q)/(vtexcoordam.p)*atlasAspect.x,1.0), vec2(1.0,(vtexcoordam.p)/(vtexcoordam.q)*atlasAspect.y));
					interval.xy *= viewCorrection;
			
			  
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
				float lum0 = luma(textureLod(texture,lmtexcoord.xy,100).rgb);
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
			gl_FragData[3].g = 1*normalTex.z;
			  lm *= normalTex.b;
			normalTex.xy = normalTex.xy*2.0-1.0;

			
			normalTex.z = sqrt(1.0 - dot(normalTex.xy, normalTex.xy));
			normalTex.z = clamp(normalTex.z,0,1);	
//			normalTex.rgb += FindNormal(texture,lmtexcoord.xy,texelSize-0.0001).rgb;
//			normalTex.rg -= FindNormal(texture,lmtexcoord.xy,texelSize-0.0003).rg;
			normal = applyBump(tbnMatrix,normalTex);
	#ifdef SSGI
	#ifdef DLM

	vec3 Q1 = dFdx(viewPos.xyz);
	vec3 Q2 = dFdy(viewPos.xyz);
	float st1 = dFdx(lm.x);
	float st2 = dFdy(lm.x);

	st1 /= dot(fwidth(viewPos.xyz), vec3(0.3));
	st2 /= dot(fwidth(viewPos.xyz), vec3(0.3));
	vec3 T = (Q1*st2 - Q2*st1);
	T = normalize(T + normal.xyz * 0.0002);
	T = cross(T, normal.xyz);

	T = normalize(T + normal * 0.01);
	T = normalize(T + normal * 0.85 * (lm.x));


	float lml = pow(clamp(dot(T, normalMat.xyz),0,1), 1.0);
	lml += pow(clamp(dot(T, normalMat.xyz) * 0.5 + 1 ,0,1), 1.0) * 0.5;
	if (dot(T, normal.xyz) > 0.99)
	{
	lml = (pow(lml, 2.0)* length(normal)*0.45) ;
	}
	lm.x = lmfaloff(lm.x);
	lm.x =  (lm.x * lml)*5;					
	lm.x =  (pow(lm.x, 0.5));
	
							
	#endif
	#endif

			data0.rgb*=color.rgb;
			vec4 data1 = clamp(noise/256.+encode(viewToWorld(normal), lm),0,1);

	

	
	#ifdef SPEC
	#ifdef labspec
			gl_FragData[1] = texture2DGradARB(specular, adjustedTexCoord.xy,dcdx,dcdy);
			gl_FragData[1].a = 0.0;
	#else	
		gl_FragData[1] = vec4(hspec.rg,hspec.b,hspec.a)/255;
			gl_FragData[1].a = 0.0;		
	#endif	

		#ifdef SPEC	
		float labemissive = textureLod(specular, adjustedTexCoord.xy, 0).a;

		float emissive = float(labemissive > 1.98 && labemissive < 2.02) * 0.25;
		float emissive2 = mix(labemissive < 1.0 ? labemissive : 0.0, 1.0, emissive);

	
	  	gl_FragData[2].a = clamp(clamp(emissive2,0.0,1.0),0,1);
	#endif
	
	
			
			#else 
			gl_FragData[1] = vec4(0.0);	
			#endif

	#else
//////////////////////////////POM-END//////////////////////////////	
	
	
	
	vec4 data0 = textureLod(texture, lmtexcoord.xy, Texture_MipMap_Bias);
	

	#ifdef SPEC
	#ifdef labspec
		gl_FragData[1] = textureLod(specular, lmtexcoord.xy, 0);
	#else	
		gl_FragData[1] = vec4(hspec.rg,hspec.b,hspec.a)/255;
	#endif	
		#else
		gl_FragData[1] = vec4(0.0);
	#endif
	#ifdef SPEC	
		float labemissive = textureLod(specular, lmtexcoord.xy, 0).a;

		float emissive = float(labemissive > 1.98 && labemissive < 2.02) * 0.25;
		float emissive2 = mix(labemissive < 1.0 ? labemissive : 0.0, 1.0, emissive);
	
	gl_FragData[2].a = clamp(clamp(emissive2,0.0,1.0),0,1);	

	#endif
				float emissivelevel = 0;
	if (mcentity == 15) emissivelevel = luma(data0.rgb);

//	gl_FragData[2].a = clamp(emissivelevel,0,1);


	data0.rgb*=color.rgb;
//    float avgBlockLum = luma(textureLod(texture, lmtexcoord.xy,128).rgb*color.rgb);
//    data0.rgb = clamp(data0.rgb*pow(avgBlockLum,-0.33)*0.85,0.0,1.0);

  
  #ifdef DISABLE_ALPHA_MIPMAPS
	data0.a = textureLod(texture,lmtexcoord.xy,0).a;
  #endif


  if (data0.a > 0.1) data0.a = normalMat.a;
  else data0.a = 0.0;


		
	#ifdef MC_NORMAL_MAP
		vec3 normalTex = textureLod(normals, lmtexcoord.xy , Texture_MipMap_Bias).rgb;
	gl_FragData[3].g = 1*normalTex.z;
		lm *= normalTex.b;
		vec2 lm2 = lm * normalTex.b;
		normalTex.xy = normalTex.xy*2.0-1.0;

		normalTex.z = sqrt(1.0 - dot(normalTex.xy, normalTex.xy));
		normalTex.z = clamp(normalTex.z,0,1);	
	
		normal = applyBump(tbnMatrix,normalTex);



	#endif


	#ifdef SSGI
	#ifdef DLM

	vec3 Q1 = dFdx(viewPos.xyz);
	vec3 Q2 = dFdy(viewPos.xyz);
	float st1 = dFdx(lm.x);
	float st2 = dFdy(lm.x);

	st1 /= dot(fwidth(viewPos.xyz), vec3(0.3));
	st2 /= dot(fwidth(viewPos.xyz), vec3(0.3));
	vec3 T = (Q1*st2 - Q2*st1);

	T = normalize(T + normal.xyz * 0.0002);		

	T = cross(T, normal.xyz);

	T = normalize(T + normal * 0.01);
	T = normalize(T + normal * 0.85 * (lm.x));


	float lml = pow(clamp(dot(T, normalMat.xyz),0,1), 1.0);
	lml += pow(clamp(dot(T, normalMat.xyz) * 0.5 + 1 ,0,1), 1.0) * 0.5;
	if (dot(T, normal.xyz) > 0.99)
	{
	lml = (pow(lml, 2.0)* length(normal)*0.5) ;
	}
	lm.x = lmfaloff(lm.x);
	lm.x =  (lm.x * lml)*5;					
	lm.x =  (pow(lm.x, 0.5));
	
							
	#endif
	#endif


	vec4 data1 = clamp(noise/256.+encode(viewToWorld(normal), lm),0.,1.0);


    vec3 scenePos   = viewMAD(gbufferModelViewInverse,fragpos);


	gl_FragData[1].a = 0.0;
	#endif	

	gl_FragData[2].rgb = normal;



	gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));


	

}
