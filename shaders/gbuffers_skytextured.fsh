

/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/

/* RENDERTARGETS: 1,10 */

in vec4 color;
in vec2 texcoord;
//faster and actually more precise than pow 2.2
vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}

uniform sampler2D texture;
void main() {

	gl_FragData[0] = texture(texture,texcoord.xy)*color;
	gl_FragData[0].rgb = gl_FragData[0].rgb*gl_FragData[0].a;

}
