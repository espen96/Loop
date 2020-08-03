

//////////////////////////////FRAGMENT//////////////////////////////		
#ifdef fsh	
//////////////////////////////FRAGMENT//////////////////////////////		



uniform sampler2D colortex4;
uniform sampler2D depthtex0;

uniform float near;
uniform float far;


float linZ(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}

//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
/* DRAWBUFFERS:4 */
	vec3 oldTex = texelFetch2D(colortex4, ivec2(gl_FragCoord.xy), 0).xyz;
	float newTex = linZ(texelFetch2D(depthtex0, ivec2(gl_FragCoord.xy*4), 0).x);
	gl_FragData[0] = vec4(oldTex, newTex);

}


#endif


//////////////////////////////VERTEX//////////////////////////////		
#ifdef vsh	
//////////////////////////////VERTEX//////////////////////////////	

	
flat varying vec3 sunColor;
flat varying vec3 moonColor;
flat varying vec3 avgAmbient;
flat varying float tempOffsets;


uniform sampler2D colortex4;
uniform int frameCounter;
#include "/lib/util.glsl"

void main() {
	tempOffsets = HaltonSeq2(frameCounter%10000);
	gl_Position = ftransform();
	gl_Position.xy = (gl_Position.xy*0.5+0.5)*clamp(0.33+0.01,0.0,1.0)*2.0-1.0;
	sunColor = texelFetch2D(colortex4,ivec2(12,37),0).rgb;
	moonColor = texelFetch2D(colortex4,ivec2(13,37),0).rgb;
	avgAmbient = texelFetch2D(colortex4,ivec2(11,37),0).rgb;

}

#endif