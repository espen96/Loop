float ld(float dist) {
    return (2.0 * near) / (far + near - dist * (far - near));
}

vec3 ssaoVL_blur(vec2 tex, vec2 dir,float cdepth)
{


	vec2 step = dir*texelSize*3.0;

  float dy = abs(dFdx(cdepth))*3.0+0.5;

	vec3 res = vec3(0.0);
	vec3 total_weights = vec3(0.);
	ivec2 pos = ivec2(gl_FragCoord.xy) % 1;
	int pixelInd = pos.y-1;
	tex += pixelInd*texelSize;
		vec3 sp = texture2D(colortex3, tex - 2.0*step).xyz;
		float linD = abs(cdepth-ld(texture2D(depthtex1,tex - 2.0*step).x)*far);
		float ssaoThresh = linD < dy*2.0 ? 1.0 : 0.;
		float weight = (ssaoThresh);
		res += sp * weight;
		total_weights += weight;

		sp = texture2D(colortex3, tex - step).xyz;
		linD = abs(cdepth-ld(texture2D(depthtex1,tex - step).x)*far);
		ssaoThresh = linD < dy ? 1.0 : 0.;
		weight = (ssaoThresh);
		res += sp * weight;
		total_weights += weight;

		sp = texture2D(colortex3, tex + step).xyz;
		linD = abs(cdepth-ld(texture2D(depthtex1,tex + step).x)*far);
		ssaoThresh = linD < dy ? 1.0 : 0.;
		weight = (ssaoThresh);
		res += sp * weight;
		total_weights += weight;

		sp = texture2D(colortex3, tex + 2.*step).xyz;
		linD = abs(cdepth-ld(texture2D(depthtex1,tex + 2.*step).x)*far);
		ssaoThresh = linD < dy*2.0 ? 1.0 : 0.;
		weight =(ssaoThresh);
		res += sp * weight;
		total_weights += weight;



		res += texture2D(colortex3, texcoord).xyz;
		total_weights += 1.;

	res /= total_weights;

	return res;
}