vec4 smoothfilter(in sampler2D tex, in vec2 uv)
{
	uv = uv*512.0 + 0.5;
	vec2 iuv = floor( uv );
	vec2 fuv = fract( uv );
	uv = iuv + (fuv*fuv)*(3.0-2.0*fuv);
	uv = uv/512.0 - 0.5/512.0;
	return texture( tex, uv);
}

float getWaterHeightmap(vec2 posxz, float iswater) {
	vec2 pos = posxz;
	float wavespeed = frameTimeCounter;
    float moving = clamp(iswater*2.-1.0,0.0,1.0);
	vec2 movement = vec2(-0.005*wavespeed*moving,0.0);
	float caustic = 0.0;
	float weightSum = 0.0;
	float radiance =  2.39996;
	mat2 rotationMatrix  = mat2(vec2(cos(radiance),  -sin(radiance)),  vec2(sin(radiance),  cos(radiance)));
	for (int i = 0; i < 4; i++){
		vec2 displ = texture(noisetex, pos/32.0/1.74/1.74 + movement).bb*2.0-1.0;
    float wave = texture(gaux3, (pos*vec2(3., 1.0)/128. + movement + displ/128.0)*exp(i*1.0)).a;
		caustic += wave*exp(-i*1.0);
		weightSum += exp(-i*1.0);
		pos = rotationMatrix * pos;
	}
	return caustic / weightSum;
}
vec3 getWaveHeight(vec2 posxz, float iswater){
	vec2 pos = posxz;
	float wavespeed = frameTimeCounter;
    float moving = clamp(iswater*2.-1.0,0.0,1.0);
	vec2 movement = vec2(-0.005*wavespeed*moving,0.0);
	vec3 caustic = vec3(0.0);
	float weightSum = 0.0;
	float radiance =  2.39996;
	mat2 rotationMatrix  = mat2(vec2(cos(radiance),  -sin(radiance)),  vec2(sin(radiance),  cos(radiance)));
	vec2 displ = texture(noisetex, pos/32.0 + movement).bb*2.0-1.0;
	for (int i = 0; i < 4; i++){
		vec2 displ = texture(noisetex, pos/32.0/1.74/1.74 + movement).bb*2.0-1.0;
    vec3 wave = texture(gaux3, (pos*vec2(3., 1.0)/128. + movement + displ/128.0)*exp(i*1.0)).rgb;
		// Hardcoded normalization
		// The python script will output these values
		wave = wave * vec3(0.28517825805472996,0.36291568757087544,0.02637002277616962) + vec3(-0.1532914212634342,-0.13959442174921308,0.9736299772192376);
		caustic += wave*exp(-i*1.0);
		weightSum += exp(-i*1.0);
		pos = rotationMatrix * pos;
	}
	return normalize(caustic / weightSum);
}
