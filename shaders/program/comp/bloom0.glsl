

//////////////////////////////FRAGMENT//////////////////////////////		
#ifdef fsh	
//////////////////////////////FRAGMENT//////////////////////////////		
uniform sampler2D colortex5;
uniform vec2 texelSize;
uniform float viewWidth;
uniform float viewHeight;

void main() {

/* DRAWBUFFERS:3 */
vec2 resScale = max(vec2(viewWidth,viewHeight),vec2(1920.0,1080.))/vec2(1920.,1080.);
vec2 quarterResTC = gl_FragCoord.xy*2.*resScale*texelSize;

		//0.5
		gl_FragData[0] = texture2D(colortex5,quarterResTC-1.0*vec2(texelSize.x,texelSize.y))/4.*0.5;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+1.0*vec2(texelSize.x,texelSize.y))/4.*0.5;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+vec2(-1.0*texelSize.x,1.0*texelSize.y))/4.*0.5;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+vec2(1.0*texelSize.x,-1.0*texelSize.y))/4.*0.5;

		//0.25
		gl_FragData[0] += texture2D(colortex5,quarterResTC-2.0*vec2(texelSize.x,0.0))/2.*0.125;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+2.0*vec2(0.0,texelSize.y))/2.*0.125;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+2.0*vec2(0,-texelSize.y))/2*0.125;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+2.0*vec2(-texelSize.x,0.0))/2*0.125;

		//0.125
		gl_FragData[0] += texture2D(colortex5,quarterResTC-2.0*vec2(texelSize.x,texelSize.y))/4.*0.125;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+2.0*vec2(texelSize.x,texelSize.y))/4.*0.125;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+vec2(-2.0*texelSize.x,2.0*texelSize.y))/4.*0.125;
		gl_FragData[0] += texture2D(colortex5,quarterResTC+vec2(2.0*texelSize.x,-2.0*texelSize.y))/4.*0.125;

		//0.125
		gl_FragData[0] += texture2D(colortex5,quarterResTC)*0.125;

		gl_FragData[0].rgb = clamp(gl_FragData[0].rgb,0.0,65000.);
		if (quarterResTC.x > 1.0 - 3.5*texelSize.x || quarterResTC.y > 1.0 -3.5*texelSize.y || quarterResTC.x < 3.5*texelSize.x || quarterResTC.y < 3.5*texelSize.y) gl_FragData[0].rgb = vec3(0.0);




}

#endif
//////////////////////////////VERTEX//////////////////////////////		
#ifdef vsh	
//////////////////////////////VERTEX//////////////////////////////	



uniform float viewWidth;
uniform float viewHeight;
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main() {
	//Improves performances and makes sure bloom radius stays the same at high resolution (>1080p)
	vec2 clampedRes = max(vec2(viewWidth,viewHeight),vec2(1920.0,1080.));
	gl_Position = ftransform();
	//*0.51 to avoid errors when sampling outside since clearing is disabled
	gl_Position.xy = (gl_Position.xy*0.5+0.5)*0.51/clampedRes*vec2(1920.0,1080.)*2.0-1.0;
}


#endif