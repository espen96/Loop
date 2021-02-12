



const ivec2 kernelO_3x3[9]  = ivec2[9](
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
    ivec2(-1, 0),  ivec2(0, 0),  ivec2(1, 0),
    ivec2(-1, 1),  ivec2(0, 1),  ivec2(1, 1)
);

const ivec2 kernelO_5x5[25] = ivec2[25] (
    ivec2(2,  2), ivec2(1,  2), ivec2(0,  2), ivec2(-1,  2), ivec2(-2,  2),
    ivec2(2,  1), ivec2(1,  1), ivec2(0,  1), ivec2(-1,  1), ivec2(-2,  1),
    ivec2(2,  0), ivec2(1,  0), ivec2(0,  0), ivec2(-1,  0), ivec2(-2,  0),
    ivec2(2, -1), ivec2(1, -1), ivec2(0, -1), ivec2(-1, -1), ivec2(-2, -1),
    ivec2(2, -2), ivec2(1, -2), ivec2(0, -2), ivec2(-1, -2), ivec2(-2, -2)
);	


vec3 screenToViewSpace(vec3 screenpos, mat4 projInv, const bool taaAware) {
    screenpos   = screenpos*2.0-1.0;



    vec3 viewpos    = vec3(vec2(projInv[0].x, projInv[1].y)*screenpos.xy + projInv[3].xy, projInv[3].z);
        viewpos    /= projInv[2].w*screenpos.z + projInv[3].w;
    
    return viewpos;
}
vec3 screenToViewSpace(vec3 screenpos, mat4 projInv) {    
    return screenToViewSpace(screenpos, projInv, true);
}

vec3 screenToViewSpace(vec3 screenpos) {
    return screenToViewSpace(screenpos, gbufferProjectionInverse);
}
vec3 screenToViewSpace(vec3 screenpos, const bool taaAware) {
    return screenToViewSpace(screenpos, gbufferProjectionInverse, taaAware);
}	
float sqr(float x) {
    return x*x;
}

const float rlog2 = 1.0/log(2.0);	
#define expf(x) exp2((x) * rlog2)

#define crcp(x) (1.0 / x)




float rcp(float x) {
    return crcp(x);
}
vec2 rcp(vec2 x) {
    return crcp(x);
}
vec3 rcp(vec3 x) {
    return crcp(x);
}


ivec2 clampTexelPos(ivec2 pos) {
    return clamp(pos, ivec2(0.0), ivec2(viewWidth,viewHeight));
}


vec2 computeVariance(sampler2D tex, ivec2 pos) {
    float sumMsqr   = 0.0;
    float sumMean   = 0.0;

    for (int i = 0; i<9; i++) {
        ivec2 deltaPos     = kernelO_3x3[i];

        vec3 col    = texelFetch(tex, clampTexelPos(pos + deltaPos), 0).rgb;
        float lum   = luma(col);

        sumMsqr    += sqr(lum);
        sumMean    += lum;
    }
    sumMsqr  /= 9.0;
    sumMean  /= 9.0;

    return vec2(abs(sumMsqr - sqr(sumMean)) * rcp(max(sumMean, 1e-20)), sumMean);
}



float hash1(float seed) {
    return fract(sin(seed)*43758.5453123);
}

#ifdef power

float Pow2(float x) { return x * x; }
vec2  Pow2(vec2  x) { return x * x; }
vec3  Pow2(vec3  x) { return x * x; }
vec4  Pow2(vec4  x) { return x * x; }
float Pow3(float x) { return x * x * x; }
float Pow4(float x) { x *= x; return x * x; }
vec2  Pow4(vec2  x) { x *= x; return x * x; }
vec3  Pow4(vec3  x) { x *= x; return x * x; }
float Pow5(float x) { float x2 = x * x; return x2 * x2 * x; }
float Pow6(float x) { x *= x; return x * x * x; }
float Pow8(float x) { x *= x; x *= x; return x * x; }
float Pow12(float x) { x *= x; x *= x; return x * x * x; }
float Pow16(float x) { x *= x; x *= x; x *= x; return x * x; }

#endif
#ifdef denoise

float blueNoise(){
  return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy)%512, 0).a + 1.0/1.6180339887 * frameCounter);
}





struct TapKey {
    float csZ;
    vec3 csPosition;
    vec3 normal;
    float glossy;
};

















uniform sampler2D colortex4;




vec3 atrous3(vec2 coord, const int size,sampler2D tex1 , float extraweight) {
    float denoiseStrength = ((DENOISE_RANGE1.x + (DENOISE_RANGE1.y-DENOISE_RANGE1.x)*hash1(641.128752*gl_FragCoord.x + 312.321374*gl_FragCoord.y+1.92357812*frameCounter)));
	int size2 = int(denoiseStrength);

    float kernel[25];
    kernel[0] = 1.0f/256.0f;
    kernel[1] = 1.0f/64.0f;
    kernel[2] = 3.0f/128.0f;
    kernel[3] = 1.0f/64.0f;
    kernel[4] = 1.0f/256.0f;
    
    kernel[5] = 1.0f/64.0f;
    kernel[6] = 1.0f/16.0f;
    kernel[7] = 3.0f/32.0f;
    kernel[8] = 1.0f/16.0f;
    kernel[9] = 1.0f/64.0f;
    
    kernel[10] = 3.0f/128.0f;
    kernel[11] = 3.0f/32.0f;
    kernel[12] = 9.0f/64.0f;
    kernel[13] = 3.0f/32.0f;
    kernel[14] = 3.0f/128.0f;
    
    kernel[15] = 1.0f/64.0f;
    kernel[16] = 1.0f/16.0f;
    kernel[17] = 3.0f/32.0f;
    kernel[18] = 1.0f/16.0f;
    kernel[19] = 1.0f/64.0f;
    
    kernel[20] = 1.0f/256.0f;
    kernel[21] = 1.0f/64.0f;
    kernel[22] = 3.0f/128.0f;
    kernel[23] = 1.0f/64.0f;
    kernel[24] = 1.0f/256.0f;
   

  

    ivec2 pos2     = ivec2(floor(coord * vec2(viewWidth, viewHeight)/RENDER_SCALE));
    float sumweight = 0.0;	
	float weight = 0.0;
	

		vec4 normaldepth = texelFetch(colortexA, pos2, 0).rgba; 


     float   c_depth    = normaldepth.a * far;	
	vec3 origNormal =  normaldepth.rgb;			
	vec3 colorCenter = texelFetch(tex1, pos2, 0).rgb; 	
	




    vec4 totalColor     = vec4(colorCenter,1);
    float totalWeight   = 1.0;	
	
    vec2 variance2  = computeVariance(colortex5, ivec2(floor(coord/RENDER_SCALE * vec2(viewWidth, viewHeight)/RENDER_SCALE)));		
    float var2        = rcp(0.5 + variance2.x *1000);	
    float sigmaL        = rcp(0.5 + variance2.x *1000);	

	
	
	float var3 = abs(variance2.x*1000);		
	
			

	if (var3 < 0.001)  return totalColor.rgb;





//#define HQ



#ifdef HQ

    for (int i = 0; i<25; i++) {
	ivec2 delta  = kernelO_5x5[i] * size;	
# else 
    for (int i = 0; i<9; i++) {
	ivec2 delta  = kernelO_3x3[i] * size;	
#endif
	
        if (delta.x == 0 && delta.y == 0) continue;
		
        ivec2 d_pos2  = pos2 + delta;
        if (clamp(d_pos2, ivec2(0), ivec2(vec2(viewWidth, viewHeight))-1) != d_pos2) continue;
        bool valid          = all(greaterThanEqual(d_pos2, ivec2(0))) && all(lessThan(d_pos2, ivec2(vec2(viewWidth, viewHeight))));
        if (!valid) continue;		

		vec4 normaldepth2 = texelFetch(colortexA, d_pos2, 0).rgba; 
        float cu_depth = (normaldepth2.a) * far;
	
		vec3 normal = (normaldepth2.rgb);			
		
		vec3 color = texelFetch(tex1, d_pos2, 0).rgb;  	
		
		float d_weight = abs(cu_depth - c_depth);	
        float depthWeight = expf(-d_weight);	
        if ((depthWeight < 1e-5 || cu_depth == 1.0)) continue;
		
	
        float normalWeight = pow(clamp(dot(normal, origNormal),0,1),64);


        float weight    = normalWeight;			
		

       
       weight *= exp(-d_weight - var2);
	   
      totalColor.rgb += color.rgb * weight;

        totalWeight += weight;
gl_FragData[1].rgb = vec3(weight);
	}

    totalColor.rgb *= rcp(max(totalWeight, 1e-25));


    return totalColor.rgb;


	
}

















#endif



vec3 edgefilter(vec2 coord, const int size,sampler2D tex1) {


  
    ivec2 pos     = ivec2(floor(coord * vec2(viewWidth, viewHeight)));
    ivec2 pos2     = ivec2(floor(coord * vec2(viewWidth, viewHeight)/RENDER_SCALE));
    float sumweight = 0.0;	
	float weight = 0.0;
	

    float c_depth  = texelFetch(depthtex0, pos2, 0).x;

        c_depth    = ld(c_depth) * far;	

	vec3 origNormal =  (texelFetch(colortexA, pos2, 0).rgb);			

    float totalWeight   = 1.0;	
	


	
    for (int i = 0; i<9; i++) {


	
        ivec2 delta  = kernelO_3x3[i];	
        if (delta.x == 0 && delta.y == 0) continue;
		
        ivec2 d_pos2  = pos2 + delta;
        if (clamp(d_pos2, ivec2(0), ivec2(vec2(viewWidth, viewHeight))-1) != d_pos2) continue;
        bool valid          = all(greaterThanEqual(d_pos2, ivec2(0))) && all(lessThan(d_pos2, ivec2(vec2(viewWidth, viewHeight))));
        if (!valid) continue;		
		
     		vec4 normaldepth2 = texelFetch(colortexA, d_pos2, 0).rgba; 
        float cu_depth = (normaldepth2.a) * far;
	
		vec3 normal = (normaldepth2.rgb);			
		
		vec3 color = texelFetch(tex1, d_pos2, 0).rgb;  	
		
		float d_weight = abs(cu_depth - c_depth);	
        float depthWeight = expf(-d_weight);	
        if ((depthWeight < 1e-5 || cu_depth == 1.0)) continue;
        float normalWeight = pow(clamp(dot(normal, origNormal),0,.9),32);


        float weight    = normalWeight;			

       
        weight *= exp(-d_weight );


        totalWeight += weight;

	
	}

    totalWeight = rcp(totalWeight)*2-1.6;


    return vec3(totalWeight)*2;


	
}


