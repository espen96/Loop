const float kernelW_3x3[9]  = float[9](
    0.0625, 0.125, 0.0625,
    0.125,  0.250, 0.125,
    0.0625, 0.125, 0.0625
);

const ivec2 kernelO_3x3[9]  = ivec2[9](
    ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
    ivec2(-1, 0),  ivec2(0, 0),  ivec2(1, 0),
    ivec2(-1, 1),  ivec2(0, 1),  ivec2(1, 1)
);
	

vec3 getDepthPoint(vec2 coord, float depth) {
    vec4 pos;
    pos.xy = coord;
    pos.z = depth;
    pos.w = 1.0;
    pos.xyz = pos.xyz * 2.0 - 1.0; //convert from the 0-1 range to the -1 to +1 range
    pos = gbufferProjectionInverse * pos;
    pos.xyz /= pos.w;
    
    return pos.xyz;
}
	
vec3 atrous(vec2 coord, const int size) {

    ivec2 pos     = ivec2(coord * vec2(viewWidth, viewHeight));
    float sumweight = 0.0;	
	float weight = 0.0;

	
    float c_depth  = texelFetch(depthtex0, pos, 0).x;
        c_depth    = ld(c_depth) * far;	
	
	
    float blur  = kernelW_3x3[4];	
	vec4 data2 = texture(colortex1, coord);
	vec4 dataUnpacked1 = vec4(decodeVec2(data2.x),decodeVec2(data2.y));			
	vec3 origNormal = decode(dataUnpacked1.yw);	

    vec3 col = vec3(0);

    for (int i = 0; i<9; i++) {
        if (i == 4) continue;	
	
        ivec2 delta  = kernelO_3x3[i] * size;	

	

        ivec2 d_pos  = pos + delta;
        if (clamp(d_pos, ivec2(0), ivec2(vec2(viewWidth, viewHeight))) != d_pos) continue;		

		vec4 data = (texelFetch(colortex1, d_pos, 0));

		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
			
		vec3 normal = decode(dataUnpacked0.yw);
        float nweight   = clamp(dot(origNormal, normal),0,1);			
        if (nweight <= 0.0) continue;			
        float cu_depth = ld(texelFetch(depthtex0, d_pos, 0).x) * far;
        if (cu_depth == 1.0) continue;		
		
        float d_weight = exp2((-max(distance(cu_depth, c_depth) - 0.075, 0.0) * 0.6) * (1.0/log(2.0)));		
        if (d_weight < 1e-5) continue;		

			

		vec3 color = texelFetch(colortex3, d_pos, 0).rgb;			
		float weight = nweight * d_weight * blur;
		col     += color * weight;
        sumweight  += 1.0 * weight;
		
	}
    col /= max(sumweight, 1e-16);

    return col;
}


vec3 atrous3(vec2 coord, int pass) {
    int kernel = 1 << pass;
	float weights = 0.0;
    float origDepth = textureLod(depthtex0, coord, 0).r;
    float origDist = getDepthPoint(coord, origDepth).z;
	
	vec4 data2 = texture(colortex1, coord).rgba;
	vec4 dataUnpacked1 = vec4(decodeVec2(data2.x),decodeVec2(data2.y));			
	vec3 origNormal = decode(dataUnpacked1.yw);	

    vec3 col = vec3(0);

	for (int i = -kernel; i <= kernel; i += 1 << pass) {
		for (int j = -kernel; j <= kernel; j += 1 << pass) {
			ivec2 icoord = ivec2(gl_FragCoord.xy) + ivec2(vec2(i,j));
			vec4 data = (texelFetch(colortex1, icoord, 0).rgba);
			vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
			vec4 dataUnpacked2 = vec4(decodeVec2(data.z),decodeVec2(data.w));
		
			vec3 normal = decode(dataUnpacked0.yw);
			
			float depth = texelFetch(depthtex0, icoord, 0).r;
			float depthDist = getDepthPoint(vec2(icoord)*vec2(viewWidth, viewHeight), depth).z;
			vec3 color = texelFetch(colortex7, icoord, 0).rgb;
			
			float weight = 1.0;
            weight *= pow(length(16 - vec2(i,j)) / 16.0, 2.0);

			weight *= pow128(clamp(dot(origNormal, normal),0,1));
			if (weight <= 0.0) continue;	
			weight *= clamp(1.0-abs(origDist - depthDist),0,1);
			if (weight == 1.0) continue;	

            if(depth == 1.0) {
                weight = 0.0;
            }
			
			col += color * weight;
			weights += weight;
		}
	}
    col = col / weights;

    return col;
}
