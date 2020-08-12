#include "/program/gbuffers/shared.glsl"


//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////


/* DRAWBUFFERS:2 */
void main() {



	gl_FragData[0] = texture2D(texture, lmtexcoord.xy)*color;
	vec2 tempOffset=offsets[framemod8];
	float exposure = texelFetch2D(gaux1,ivec2(10,37),0).r;		
	float torch_lightmap = lmtexcoord.z;	
	vec3 normal = normalMat.xyz;
	vec4 albedo = vec4(0.0);
	#ifdef spidereye
	 albedo = texture2D(texture, texcoord);
	 albedo.rgb =srgbToLinear(albedo.rgb);
	#else 
	  albedo.rgb = srgbToLinear(gl_FragData[0].rgb);
	#endif
	vec3 col = albedo.rgb/exposure*0.1;	
		
		

	vec3 fragpos = toScreenSpace(gl_FragCoord.xyz*vec3(texelSize/RENDER_SCALE,1.0)-vec3(vec2(tempOffset)*texelSize*0.5,0.0));
		float NdotL = lightCol.a*dot(normal,sunVec);	

	vec4 final = vec4(0.0);		


	#ifdef textured
			  NdotL = -lightSign*dot(normal,sunVec);
	#endif	  
		
		float NdotU = dot(upVec,normal);
		float diffuseSun = clamp(NdotL,0.0f,1.0f);
	#ifdef textured
			  diffuseSun = 0.712;
	#endif	  
		vec3 direct = texelFetch2D(gaux1,ivec2(6,37),0).rgb/3.1415;


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
				const float threshMul = sqrt(2048.0/shadowMapResolution*shadowDistance/128.0);
				float distortThresh = 1.0/(distortFactor*distortFactor);
				float diffthresh = facos(diffuseSun)*distortThresh/800*threshMul;
		#ifdef textured
					  diffthresh = 0.0002;
		#endif	

				projectedShadowPosition = projectedShadowPosition * vec3(0.5,0.5,0.5/6.0) + vec3(0.5,0.5,0.5);

				float noise = interleaved_gradientNoise(tempOffset.x*0.5+0.5);


				vec2 offsetS = vec2(cos( noise*TAU ),sin( noise*TAU ));

				float shading = shadow2D_bicubic(shadow,vec3(projectedShadowPosition + vec3(0.0,0.0,-diffthresh*1.2)));


				direct *= shading;
			}

		}


		direct *= diffuseSun;
		vec3 ambient = texture2D(gaux1,(lmtexcoord.zw*15.+0.5)*texelSize).rgb;
		
		
		vec3 diffuseLight = direct + ambient;
		
		gl_FragData[0].rgb = diffuseLight*albedo.rgb*8./1500.*0.1;
		

#ifdef beaconbeam

		diffuseLight = torch_lightmap*vec3(20.,30.,50.)*2./10. ;
	    gl_FragData[0].a += 0.75;
		gl_FragData[0].rgb = (diffuseLight*albedo.rgb/exposure*5.0)*0.01;
		

#endif
		
#ifdef textured
		
		diffuseLight = direct*lmtexcoord.w + ambient;
		gl_FragData[0].rgb = diffuseLight*albedo.rgb*8./3.0/150.*0.1;	
		
#endif

#ifdef weather

		gl_FragData[0].a = clamp(gl_FragData[0].a -0.1,0.0,1.0)*0.5;
		gl_FragData[0].rgb = dot(albedo.rgb,vec3(1.0))*ambient*10./3.0/150.*0.1;
	
#endif

#ifdef glint	

		gl_FragData[0].rgb = col*color.a*0.05;
		gl_FragData[0].a = 0.2;	

#endif

#ifdef spidereye	
	
	float alblum = luma(albedo.rgb);
	if (alblum <=0) albedo.a =0;
	albedo.rgb *= albedo.a;
	gl_FragData[0].rgb = albedo.rgb;
	gl_FragData[0].a = albedo.a;
	
#endif
#ifdef thand	
	

		gl_FragData[0].rgb = diffuseLight*albedo.rgb*8./1500.;

	
#endif

	gl_FragData[1].r = 0.1;	
	
	
}

