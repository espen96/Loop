
#include "/program/gbuffers/shared.glsl"

//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

/* DRAWBUFFERS:132 */
	void main() {
	
#if MC_VERSION >= 11500 && defined TEMPORARY_FIX
#undef MC_NORMAL_MAP
#undef PBR
#endif


	float lightningBolt = float(entityId == 58);	

	vec2 tempOffset=offsets[framemod8];

	
	float noise = interleaved_gradientNoise();
	vec3 normal = normalMat.xyz;
	vec4 specularity = vec4(0.0);
	vec3 mat_data = vec3(0.0);
	vec4 reflected = vec4(0.0);

	vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize/RENDER_SCALE,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));
	vec3 p3 = mat3(gbufferModelViewInverse) * fragpos + gbufferModelViewInverse[3].xyz;



	vec4 color = color;	
	


#ifdef MC_NORMAL_MAP
		vec3 tangent2 = normalize(cross(tangent.rgb,normal)*tangent.w);
		mat3 tbnMatrix = mat3(tangent.x, tangent2.x, normal.x,
								  tangent.y, tangent2.y, normal.y,
						     	  tangent.z, tangent2.z, normal.z);
#endif

//////////////////////////////PBR//////////////////////////////



#ifdef PBR		
	float ao = 1.0;
	color.rgb *= ao;
	vec3 direct = texelFetch2D(gaux1,ivec2(6,37),0).rgb/3.1415;	
	float iswater = normalMat.w;		
	float NdotL = lightSign*dot(normal,sunVec);
	float NdotU = dot(upVec,normal);


	
	float diffuseSun = clamp(NdotL,0.0f,1.0f);	
	float shading = 0.0;
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
					float rdMul = 4.0/shadowMapResolution;
					mat2 noiseM = mat2( cos( noise*3.14159265359*2.0 ), -sin( noise*3.14159265359*2.0 ),
										 sin( noise*3.14159265359*2.0 ), cos( noise*3.14159265359*2.0 )
										);
					for(int i = 0; i < 6; i++){
						vec2 offsetS = noiseM*shadowOffsets[i];

						float weight = 1.0+(i+noise)*rdMul/SHADOW_FILTER_SAMPLE_COUNT*shadowMapResolution;
						shading += shadow2D(shadow,vec3(projectedShadowPosition + vec3(rdMul*offsetS,-diffthresh*weight))).x/6.0;
					}

					direct *= shading;


			}

		}	
		
		direct *= diffuseSun*lmtexcoord.w;
		shading = mix(0.0, shading, clamp(eyeBrightnessSmooth.y/255.0 + lmtexcoord.w,0.0,1.0));		
		vec3 diffuseLight = direct + texture2D(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize).rgb;

		vec3 linao = LinearTosRGB(texture2D(normals, lmtexcoord.xy).zzz);
		ao = clamp(linao.z,0,1);


		vec2 adjustedTexCoord = fract(vtexcoord.st)*vtexcoordam.pq+vtexcoordam.st;

		
		
		
#ifdef POM2		


		vec3 viewVector = normalize(tbnMatrix*fragpos);
		float dist = length(fragpos);


    if (dist < MAX_OCCLUSION_DISTANCE) {
    	#ifndef USE_LUMINANCE_AS_HEIGHTMAP
    		if ( viewVector.z < 0.0 && readNormal(vtexcoord.st).a < 0.9999 && readNormal(vtexcoord.st).a > 0.00001) {
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
    	#else
    	if ( viewVector.z < 0.0)
    	{
    		vec3 interval = viewVector.xyz * intervalMult;
    		vec3 coord = vec3(vtexcoord.st, 1.0);
    		coord += noise*interval;
        float lum0 = luma(texture2DLod(texture,lmtexcoord.xy,100).rgb);
    		for (int loopCount = 0;
    				(loopCount < MAX_OCCLUSION_POINTS) && (luma(readTexture(coord.st).rgb)/lum0 < coord.p) &&coord.p >= 0.0;
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
	
	
	#ifdef  MC_NORMAL_MAP
	
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
	#endif
	
	
#ifdef PBR
	vec4 alb = texture2DGradARB(texture, adjustedTexCoord.xy,dcdx,dcdy);
	specularity = texture2DGradARB(specular, adjustedTexCoord, dcdx, dcdy).rgba;
	bool is_metal = false;
	mat_data = labpbr(specularity, is_metal);
		float roughness = mat_data.x;

		float emissive = 0.0;
		float F0 = mat_data.y;
		vec3 reflectedVector = reflect(normalize(fragpos), normal);
		float normalDotEye = dot(normal, normalize(fragpos));
		float fresnel = pow(clamp(1.0 + normalDotEye,0.0,1.0), 5.0);
		fresnel = mix(F0,1.0,fresnel);
		vec3 wrefl = mat3(gbufferModelViewInverse)*reflectedVector;

		vec3 sky_c = mix(skyCloudsFromTex(wrefl,gaux1).rgb,texture2D(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize).rgb*0.5,isEyeInWater);
		     sky_c.rgb *= lmtexcoord.w*lmtexcoord.w*255*255/240.0/240.0/150.0*fresnel/3.0;

		vec4 reflection = vec4(0.0);
		     reflection.rgb = mix(sky_c.rgb*0.1, clamp(reflection.rgb,0,5), reflection.a);

			float spec = (F0/255)*SPEC_MULTIPLIER;
		//	vec3 sunSpec = GGX(normal,normalize(fragpos),  lightSign*sunVec, mat_data.xy+0.02)* (texelFetch2D(gaux1,ivec2(6,37),0).rgb*spec)*10.0/3./150.0/3.1415 * (1.0-rainStrength*0.9)*clamp(sunElevation,0.01,1);
			vec3 sunSpec = GGX(normal,normalize(fragpos),  lightSign*sunVec, mat_data.xy+0.02)* (texelFetch2D(gaux1,ivec2(6,37),0).rgb*spec)*8./3./150.0/3.1415 * (1.0-rainStrength*0.9);				
			


		vec3 sp = (reflection.rgb*fresnel+shading*sunSpec);


		if (is_metal) {
			if (F0 < 1.0) sp.rgb *= MetalCol(F0);	
				else sp *= alb.rgb;			
			reflected.rgb += sp;
			
		} else {
			reflected.rgb += sp ;
		}
		
		reflected *= shading;
#endif
	

    vec4 data0 = vec4(texture2DGradARB(texture, adjustedTexCoord.xy,dcdx,dcdy).rgb,texture2DGradARB(texture, adjustedTexCoord.xy,vec2(0.),vec2(0.0)).a);
	
	
	
	     data0.rgb = mix(data0.rgb,entityColor.rgb,entityColor.a);
#ifdef hand 	
	if (data0.a > 0.1) data0.a = normalMat.a*0.5+0.5;
#else
	if (data0.a > 0.1) data0.a = normalMat.a;
#endif
	else data0.a = 0.0;
	data0.rgb*=color.rgb;
	

	

	#ifdef entities
	normal = normalMat.xyz;
	data0 = texture2D(texture, lmtexcoord.xy)*color;
	data0.rgb = mix(data0.rgb,entityColor.rgb,entityColor.a);
	
	
	
	if (data0.a > 0.3) data0.a = normalMat.a*0.5+0.1;
	else data0.a = 0.0;
	#endif	

	vec4 data1 = clamp(encode(normal),0.,1.0);
	#else
	
	
  vec4 data0 = texture2D(texture, lmtexcoord.xy, Texture_MipMap_Bias);
  data0.rgb*=color.rgb;
  float avgBlockLum = luma(texture2DLod(texture, lmtexcoord.xy,128).rgb*color.rgb);
  data0.rgb = clamp(data0.rgb*pow(avgBlockLum,-0.33)*0.85,0.0,1.0);



  #ifdef DISABLE_ALPHA_MIPMAPS
  data0.a = texture2DLod(texture,lmtexcoord.xy,0).a;
  #endif



#ifdef hand 	
	if (data0.a > 0.1) data0.a = normalMat.a*0.5+0.5;
#else
	if (data0.a > 0.1) data0.a = normalMat.a;
#endif
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
	#ifdef entities
	normal = normalMat.xyz;

	data0.rgb = texture2D(texture, lmtexcoord.xy).rgb*color.rgb;
	
	data0.rgb = mix(data0.rgb,entityColor.rgb,entityColor.a);



	
	
	if (data0.a > 0.3) data0.a = normalMat.a*0.5+0.1;
	else data0.a = 0.0;
	

#endif
	
	vec4 data1 = clamp(noise/256.0+encode(normal),0.,1.0);



#endif	
	
	
vec2 spec1 =specularity.rg;
vec2 spec2 = vec2(specularity.b,mat_data.z);
		
		
#ifndef entity
	gl_FragData[2] = clamp(vec4(reflected.rgb,0),0.0,10.0);
	gl_FragData[1].a = 0;
	gl_FragData[1].gb = vec2(encodeVec2v2(spec1),encodeVec2v2(spec2));

#endif	

	
#ifdef entity 	
gl_FragData[2].rgb = vec3(0.0);
#endif	

#ifdef basic 	
gl_FragData[2].rgb = vec3(0.0);
#endif	

#ifndef block
gl_FragData[1].a = 0.1;	
#endif

if (lightningBolt > 0.5) data0.rgb = vec3(1.0), data0.a = 0.5;	
if (lightningBolt > 0.5) data1 = vec4(1.0);	
if (lightningBolt > 0.5) gl_FragData[1].a = 0.2;	



gl_FragData[0] = vec4(encodeVec2(data0.x,data1.x),encodeVec2(data0.y,data1.y),encodeVec2(data0.z,data1.z),encodeVec2(data1.w,data0.w));			



}
