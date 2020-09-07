# define spidereye
#include "/program/gbuffers/shared.glsl"

uniform float viewWidth;
uniform float viewHeight;




void main(){

	float exposure = texelFetch2D(gaux1,ivec2(10,37),0).r;		

    vec4 albedo = texture2D(texture, lmtexcoord.xy)*color;
	
	
	
	
	
	

	albedo.rgb = albedo.rgb/exposure*0.1;
        albedo.rgb = albedo.rgb;
    /* DRAWBUFFERS:2 */
	
	
    gl_FragData[0] = albedo;
}

