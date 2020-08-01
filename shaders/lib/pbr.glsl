float cdist(vec2 coord){
	return max(abs(coord.s-0.5),abs(coord.t-0.5))*1.85;
}

float pow2(float x) {
    return x*x;
}	
float pow3(float x){
    return x*x*x;
}
float pow4(float x){
    return pow2(pow2(x));
}
float pow5(float x){
    return pow4(x)*x;
}

float linStep(float x, float low, float high) {
    float t = clamp(((x-low)/(high-low)),0,1);
    return t;
}
	
vec3 nvec3(vec4 pos){
    return pos.xyz/pos.w;
}
vec4 nvec4(vec3 pos){
    return vec4(pos.xyz, 1.0);
}


float GGX (vec3 n, vec3 v, vec3 l, vec2 material) {
    float F0  = material.y;
    float r = pow2(material.x);


  vec3 h = l - v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = (dot(h,l)*hn);
  float dotNH = (dot(h,n)*hn);
  float dotNL = (dot(n,l));

  float denom = dotNH * r - dotNH + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  float F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}








vec3 MetalCol(float f0){
    int metalidx = int(f0 * 255.0);

    if (metalidx == 230) return vec3(0.24867, 0.22965, 0.21366); //iron
    if (metalidx == 231) return vec3(0.88140, 0.57256, 0.11450); //gold
    if (metalidx == 232) return vec3(0.81715, 0.82021, 0.83177); //aluminium
    if (metalidx == 233) return vec3(0.27446, 0.27330, 0.27357); //chrome
    if (metalidx == 234) return vec3(0.84430, 0.48677, 0.22164); //copper
    if (metalidx == 235) return vec3(0.36501, 0.35675, 0.37653); //lead
    if (metalidx == 236) return vec3(0.42648, 0.37772, 0.31138); //platinum
    if (metalidx == 237) return vec3(0.91830, 0.89219, 0.83662); //silver
    return vec3(1.0);
}									
									
vec3 labpbr(vec4 unpacked_tex, out bool is_metal) {
	vec3 mat_data = vec3(1.0, 0.0, 0.0);

    mat_data.x  = pow2(1.0 - unpacked_tex.x);   //roughness
    mat_data.y  = (unpacked_tex.y);         //f0

    unpacked_tex.w = unpacked_tex.w * 255.0;

    mat_data.z  = unpacked_tex.w < 254.5 ? linStep(unpacked_tex.w, 0.0, 254.0) : 0.0; //emission

    is_metal    = (unpacked_tex.y * 255.0) > 229.5;


	return mat_data;
}	


vec3 sspr(vec3 dir,vec3 position,float noise, float fresnel){
	
	vec3 clipPosition = toClipSpace3(position)*vec3(RENDER_SCALE,1.0);
	

	
	
	float stepSize = 15;
	int maxSteps = 15;	
	int maxLength = 15;	
	
	


	float rayLength = ((position.z + dir.z * sqrt(3.0)*maxLength) > -sqrt(3.0)*near) ?  (-sqrt(3.0)*near -position.z) / dir.z : sqrt(3.0)*maxLength;

	vec3 end = toClipSpace3(position+dir*rayLength)*vec3(RENDER_SCALE,1.0);
	vec3 direction = end-clipPosition;  //convert to clip space
	
	
	float len = max(abs(direction.x)/texelSize.x,abs(direction.y)/texelSize.y)/stepSize;
	
	
	//get at which length the ray intersects with the edge of the screen
	vec3 maxLengths = (step(0.,direction)-clipPosition) / direction;
	float mult = min(min(maxLengths.x,maxLengths.y),maxLengths.z);


	vec3 stepv = direction/len;

	
	int iterations = min(int(min(len, mult*len)-2), maxSteps);	

	//Do one iteration for closest texel (good contact shadows)
	vec3 spos = clipPosition + stepv/stepSize*4.0;
	spos.xy+= TAA_Offset*texelSize*0.5;
	
	float sp = texelFetch2D(colortex4,ivec2(spos.xy/texelSize/4),0).w;
	float currZ = linZ(spos.z);


	spos += stepv*noise;	
	
 for(int i = 0; i < iterations; i++){
		float sp = texelFetch2D(colortex4,ivec2(spos.xy/texelSize/4),0).w;
		float currZ = linZ(spos.z);
		if( sp < currZ) {
			float dist = abs(sp-currZ)/currZ;
			if (dist <= 0.035 ) return vec3(spos.xy, invLinZ(sp))/vec3(RENDER_SCALE,1.0);
		}
			spos += stepv;

	}
	
    return vec3(1.1);
}
vec3 cosineHemisphereSample2(vec2 Xi)
{
    float r = sqrt(Xi.x);
    float theta = 2.0 * 3.14159265359 * Xi.y;

    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec3(x, y, sqrt(max(0.0f, 1 - Xi.x)));
}

vec2 R2_samples2(int n){
	vec2 alpha = vec2(0.75487765, 0.56984026);
	return fract(alpha * n);
}

vec3 TangentToWorld2(vec3 N, vec3 H)
{
    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(UpVector, N));
    vec3 B = cross(N, T);

    return vec3((T * H.x) + (B * H.y) + (N * H.z));
}

vec3 SSPTR(vec3 normal,vec4 noise,vec3 fragpos,float roughness, float f0, float fresnel){
	int nrays = 1;
	vec3 intRadiance = vec3(0.0);
	for (int i = 0; i < nrays; i++){

	
	//if (roughness>=0.6) break;

	

	
		int seed = (frameCounter%10000)*nrays+i;
		vec2 ij = fract(R2_samples2(seed) + noise.rg);
		vec3 rayDir = normalize(cosineHemisphereSample2(ij));
		rayDir = TangentToWorld2(normal*fresnel,rayDir);
		vec3 offset = rayDir-normal;
	if (offset.x>=0.6) break;

		vec3 reflectedVector = reflect(normalize(fragpos), normalize(mix(normal,rayDir,(roughness))));
		
		vec3 rayHit = sspr(reflectedVector, fragpos, fract(seed/1.6180339887),fresnel);

		if (rayHit.z < 1.){
			vec3 previousPosition = mat3(gbufferModelViewInverse) * toScreenSpace(rayHit) + gbufferModelViewInverse[3].xyz + cameraPosition-previousCameraPosition;
			previousPosition = mat3(gbufferPreviousModelView) * previousPosition + gbufferPreviousModelView[3].xyz;
			previousPosition.xy = projMAD(gbufferPreviousProjection, previousPosition).xy / -previousPosition.z * 0.5 + 0.5;
			if (previousPosition.x > 0.0 && previousPosition.y > 0.0 && previousPosition.x < 1.0 && previousPosition.x < 1.0)



			
			intRadiance = mix(texture2D(colortex5,previousPosition.xy).rgb,texture2D(colortex3,rayHit.xy*RENDER_SCALE).rgb,0.75).rgb;
				
		
		} 
//	else { intRadiance += clamp(sky,0,10);}

	}
	return clamp(intRadiance/nrays,0,1);
}
