float square(float x){
  return x*x;
}

float g(float NdotL, float roughness)
{
    float alpha = square(max(roughness, 0.02));
    return 2.0 * NdotL / (NdotL + sqrt(square(alpha) + (1.0 - square(alpha)) * square(NdotL)));
}

float gSimple(float dp, float roughness){
  float k = roughness + 1;
  k *= k/8.0;
  return dp / (dp * (1.0-k) + k);
}

vec3 GGX2(vec3 n, vec3 v, vec3 l, float r, vec3 F0) {
  float alpha = square(r);

  vec3 h = normalize(l + v);

  float dotLH = clamp(dot(h,l),0.,1.);
  float dotNH = clamp(dot(h,n),0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNV = clamp(dot(n,v),0.,1.);
  float dotVH = clamp(dot(h,v),0.,1.);


  float D = alpha / (3.141592653589793*square(square(dotNH) * (alpha - 1.0) + 1.0));
  float G = gSimple(dotNV, r) * gSimple(dotNL, r);
  vec3 F = F0 + (1. - F0) * exp2((-5.55473*dotVH-6.98316)*dotVH);

  return dotNL * F * (G * D / (4 * dotNV * dotNL + 1e-7));
}
float invLinZ (float lindepth){
	return -((2.0*near/lindepth)-far-near)/(far-near);
}

vec3 GGX (vec3 n, vec3 v, vec3 l, float r, vec3 F0) {
  r*=r;r*=r;

  vec3 h = l + v;
  float hn = inversesqrt(dot(h, h));

  float dotLH = clamp(dot(h,l)*hn,0.,1.);
  float dotNH = clamp(dot(h,n)*hn,0.,1.);
  float dotNL = clamp(dot(n,l),0.,1.);
  float dotNHsq = dotNH*dotNH;

  float denom = dotNHsq * r - dotNHsq + 1.;
  float D = r / (3.141592653589793 * denom * denom);
  vec3 F = F0 + (1. - F0) * exp2((-5.55473*dotLH-6.98316)*dotLH);
  float k2 = .25 * r;

  return dotNL * D * F / (dotLH*dotLH*(1.0-k2)+k2);
}
vec3 toClipSpace3(vec3 viewSpacePosition) {
    return projMAD(gbufferProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}
vec3 rayTrace(vec3 dir,vec3 position,float dither, float quality){

    vec3 clipPosition = toClipSpace3(position);
  	float rayLength = ((position.z + dir.z * far*sqrt(3.)) > -near) ? (-near -position.z) / dir.z : far*sqrt(3.);
    vec3 direction = normalize(toClipSpace3(position+dir*rayLength)-clipPosition);  //convert to clip space
    direction.xy = normalize(direction.xy);

    //get at which length the ray intersects with the edge of the screen
    vec3 maxLengths = (step(0.,direction)-clipPosition) / direction;
    float mult = min(min(maxLengths.x,maxLengths.y),maxLengths.z);


    vec3 stepv = direction * mult / quality*vec3(RENDER_SCALE,1.0);




	vec3 spos = clipPosition*vec3(RENDER_SCALE,1.0) + stepv*dither;
	float minZ = clipPosition.z+stepv.z*clamp(dither-0.5,0.0,1.0);
	float maxZ = spos.z+stepv.z*(0.5+dither);
	spos.xy += TAA_Offset*texelSize*0.5/RENDER_SCALE;

  for (int i = 0; i <= int(quality); i++) {
    if (spos.x < 0.0 && spos.y < 0.0 && spos.z < 0.0 && spos.x > 1.0 && spos.y > 1.0 && spos.z > 1.0)
      return vec3(1.1);
		// decode depth buffer
		float sp = sqrt(texelFetch2D(colortex4,ivec2(spos.xy/texelSize/4),0).w/65000.0);
    if(sp <= ld(spos.z) && abs(sp-ld(spos.z))/ld(spos.z) < 0.1){
			return vec3(spos.xy/RENDER_SCALE,spos.z);
    }
    spos += stepv;
  }
  return vec3(1.1);
}
void frisvad(in vec3 n, out vec3 f, out vec3 r){
    if(n.z < -0.999999) {
        f = vec3(0.,-1,0);
        r = vec3(-1, 0, 0);
    } else {
    	float a = 1./(1.+n.z);
    	float b = -n.x*n.y*a;
    	f = vec3(1. - n.x*n.x*a, b, -n.x);
    	r = vec3(b, 1. - n.y*n.y*a , -n.y);
    }
}
mat3 CoordBase(vec3 n){
	vec3 x,y;
    frisvad(n,x,y);
    return mat3(x,y,n);
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
	


vec3 sampleGGXVNDF(vec3 V_, float alpha_x, float alpha_y, float U1, float U2){
	// stretch view
	vec3 V = normalize(vec3(alpha_x * V_.x, alpha_y * V_.y, V_.z));
	// orthonormal basis
	vec3 T1 = (V.z < 0.9999) ? normalize(cross(V, vec3(0,0,1))) : vec3(1,0,0);
	vec3 T2 = cross(T1, V);
	// sample point with polar coordinates (r, phi)
	float a = 1.0 / (1.0 + V.z);
	float r = sqrt(U1);
	float phi = (U2<a) ? U2/a * 3.141592653589793 : 3.141592653589793 + (U2-a)/(1.0-a) * 3.141592653589793;
	float P1 = r*cos(phi);
	float P2 = r*sin(phi)*((U2<a) ? 1.0 : V.z);
	// compute normal
	vec3 N = P1*T1 + P2*T2 + sqrt(max(0.0, 1.0 - P1*P1 - P2*P2))*V;
	// unstretch
	N = normalize(vec3(alpha_x*N.x, alpha_y*N.y, max(0.0, N.z)));
	return N;
}

float unpackRoughness(float x){
  float r = 1.0 - x;
  return r*r;
}
