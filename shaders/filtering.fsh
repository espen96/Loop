uniform sampler2D colortex1;
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex7;

uniform sampler2D colortex10;
uniform sampler2D colortex11;
uniform sampler2D colortex12;
uniform sampler2D colortex13;
uniform sampler2D colortex14;
uniform sampler2D colortex15;
uniform sampler2D colortex8;
uniform sampler2D colortex9;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferPreviousModelView;

uniform sampler2D noisetex;//depth
uniform int frameCounter;
flat in vec2 TAA_Offset;
uniform vec2 texelSize;
uniform vec2 viewSize;
uniform float frameTimeCounter;
uniform float viewHeight;
uniform float viewWidth;
uniform vec3 previousCameraPosition;

#define fsign(a)  (clamp((a)*1e35,0.,1.)*2.-1.)
#define denoise
#define power
#include "/lib/res_params.glsl"
#include "/lib/color_transforms.glsl"
#include "/lib/projections.glsl"
#include "/lib/kernel.glsl"
uniform float far;
uniform float near;
uniform float aspectRatio;

vec2 texcoord = gl_FragCoord.xy * texelSize;

vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
	return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

float ld(float dist) {
	return (2.0 * near) / (far + near - dist * (far - near));
}

vec2 decodeVec2(float a) {
	const vec2 constant1 = 65535. / vec2(256., 65536.);
	const float constant2 = 256. / 255.;
	return fract(a * constant1) * constant2;
}

#define DENOISE_RANGE1 vec2(32, 30)

#include "/lib/filter.glsl"

void main() {

/* RENDERTARGETS: 8 */

	float z = texture(depthtex1, texcoord).x;
	vec4 data = texture(colortex1, texcoord);
	vec4 dataUnpacked1 = vec4(decodeVec2(data.z), decodeVec2(data.w));
	bool hand = abs(dataUnpacked1.w - 0.75) < 0.01;
	bool emissive = abs(dataUnpacked1.w - 0.9) < 0.01;
	bool iswater = texture(colortex7, texcoord).a > 0.99;
	vec4 color = texture(colortex8, texcoord).rgba;		

#ifdef ssptfilter
	if (FILTER_STEPS >= FILTER_STAGE && !emissive) {

		if (z < 1 && !iswater && !hand)
			color.rgb = clamp(atrous3(texcoord.xy * RENDER_SCALE, DENOISE_RANGE, colortex8, 0.0).rgb, 0, 10);

	}
#endif

	gl_FragData[0] = color;

}
