float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}


		
		
vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}		


vec3 ssaoVL_blur(vec2 tex, vec2 dir,float cdepth)
{
	vec3 closestToCamera = vec3(texcoord,texture2D(depthtex2,texcoord).x);
	vec3 fragposition = toScreenSpace(closestToCamera);
	fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz + (cameraPosition - previousCameraPosition);
	vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
	previousPosition = toClipSpace3Prev(previousPosition);
	vec2 velocity = previousPosition.xy - closestToCamera.xy;


    float vel = abs(velocity.x+velocity.y)*100;	

	vec2 step = dir*texelSize*clamp((0.0),0,16);

  float dy = abs(dFdx(cdepth))*3.0+1;
if(emissive) step = vec2(0.0);
	vec3 res = vec3(0.0);
	vec3 total_weights = vec3(0.);
	ivec2 pos = ivec2(gl_FragCoord.xy) % 1;
	int pixelInd = pos.y;
	tex += pixelInd*texelSize;
		vec3 sp = texture2D(colortex3, tex - 2*step).xyz;
		float linD = abs(cdepth-ld(texture2D(depthtex1,tex - 2.0*step).x)*far);
		float ssaoThresh = linD < dy*2.0 ? 1.0 : 0.;
		float weight = (ssaoThresh);
		res += sp * weight;
		total_weights += weight;
		


		sp = texture2D(colortex3, tex + 2*step).xyz;
		linD = abs(cdepth-ld(texture2D(depthtex1,tex + 2.*step).x)*far);
		ssaoThresh = linD < dy*2.0 ? 1.0 : 0.;
		weight =(ssaoThresh);
		res += sp * weight;
		total_weights += weight;



		res += texture2D(colortex3, texcoord).xyz;
		total_weights += 1.0;

	res /= total_weights;

	return res;
}

