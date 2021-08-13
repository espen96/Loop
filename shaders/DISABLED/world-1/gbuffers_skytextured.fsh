#version 150

/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/

/* RENDERTARGETS: 1 */

varying vec4 color;
varying vec2 texcoord;
//faster and actually more precise than pow 2.2
vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}
     uniform int renderStage; 

// 0 Undefined
// 1  Sky
// 2  Sunset and sunrise overlay
// 3  Custom sky
// 4  Sun
// 5  Moon
// 6  Stars
// 7  Void
// 8  Terrain solid
// 9  Terrain cutout mipped
// 10 Terrain cutout
// 11 Entities
// 12 Block entities
// 13 Destroy overlay
// 14 Selection outline
// 15 Debug renderers
// 16 Solid handheld objects
// 17 Terrain translucent
// 18 Tripwire string
// 19 Particles
// 20 Clouds
// 21 Rain and snow
// 22 World border
// 23 Translucent handheld objects

uniform sampler2D texture;
void main() {
if(renderStage != 5) discard;
	gl_FragData[0] = texture(texture,texcoord.xy)*color;
	gl_FragData[0].rgb = gl_FragData[0].rgb*gl_FragData[0].a;

}
