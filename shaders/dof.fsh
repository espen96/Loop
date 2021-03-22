#version 130
//Vignetting, applies bloom, applies exposure and tonemaps the final image
#extension GL_EXT_gpu_shader4 : enable
#define Fake_purkinje
//#define motionblur

#define BLOOMY_FOG 0.5 //[0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 3.0 4.0 6.0 10.0 15.0 20.0]
#define BLOOM_STRENGTH  1.0 //[0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 3.0 4.0]
#define TONEMAP Tonemap_Loop // Tonemapping operator [ HableTonemap reinhard Tonemap_Lottes ACESFilm ToneMap_Hejl2015]
//#define USE_ACES_COLORSPACE_APPROXIMATION	// Do the tonemap in another colorspace

#define Purkinje_strength 1.0	// Simulates how the eye is unable to see colors at low light intensities. 0 = No purkinje effect at low exposures [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]

#define Purkinje_R 0.4 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Purkinje_G 0.7 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]
#define Purkinje_B 1.0 // [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0]



#define Purkinje_Multiplier 5.0 // How much the purkinje effect increases brightness [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2.0 2.05 2.1 2.15 2.2 2.25 2.3 2.35 2.4 2.45 2.5 2.55 2.6 2.65 2.7 2.75 2.8 2.85 2.9 2.95 3.0 3.05 3.1 3.15 3.2 3.25 3.3 3.35 3.4 3.45 3.5 3.55 3.6 3.65 3.7 3.75 3.8 3.85 3.9 3.95 4.0 4.05 4.1 4.15 4.2 4.25 4.3 4.35 4.4 4.45 4.5 4.55 4.6 4.65 4.7 4.75 4.8 4.85 4.9 4.95 5.0 5.05 5.1 5.15 5.2 5.25 5.3 5.35 5.4 5.45 5.5 5.55 5.6 5.65 5.7 5.75 5.8 5.85 5.9 5.95 6.0 6.05 6.1 6.15 6.2 6.25 6.3 6.35 6.4 6.45 6.5 6.55 6.6 6.65 6.7 6.75 6.8 6.85 6.9 6.95 7.0 7.05 7.1 7.15 7.2 7.25 7.3 7.35 7.4 7.45 7.5 7.55 7.6 7.65 7.7 7.75 7.8 7.85 7.9 7.95 8.0 8.05 8.1 8.15 8.2 8.25 8.3 8.35 8.4 8.45 8.5 8.55 8.6 8.65 8.7 8.75 8.8 8.85 8.9 8.95 9.0 9.05 9.1 9.15 9.2 9.25 9.3 9.35 9.4 9.45 9.5 9.55 9.6 9.65 9.7 9.75 9.8 9.85 9.9 9.95 ]


//#define DOF							//enable depth of field (blur on non-focused objects)
//					//Slow! Forces circular bokeh!  Uses 4 times more samples with noise in order to remove sampling artifacts at great blur sizes.
//		//disabled : circular blur shape - enabled : hexagonal blur shape
#define EXCLUDE_MODE 0 // [0 1 2]
#define BOKEH_MODE 0 // [0 1 2 3]
#define DOF_MODE 0 // [0 1 2]

//#define FAR_BLUR_ONLY // Removes DoF on objects closer to the camera than the focus point
//lens properties
#define focal 24 // Lens focal length in millimeters [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300]
#define aperture 16.0 // Aperture size in f-stops. Larger number = smaller aperture. [0.8 1.4 1.8 2.0 2.8 3.6 4.0 5.6 8.0 11.0 16.0 22.0]
#define MANUAL_FOCUS 48.0 // If autofocus is turned off, sets the focus point (meters)	[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 5.0 6.0 7.0 8.0 10.0 12.0 14.0 16.0 20.0 24.0 28.0 32.0 40.0 48.0 56.0 64.0 92.0 128.0 192.0 256.0 384.0 512.0]
#include "/lib/res_params.glsl"
#include "/lib/kernel.glsl"




flat varying vec4 exposure;
flat varying vec2 rodExposureDepth;
varying vec2 texcoord;
uniform sampler2D colortex4;
uniform sampler2D colortex2;
uniform sampler2D colortex5;
uniform sampler2D colortex3;
uniform sampler2D colortex6;
uniform sampler2D colortex7;
uniform sampler2D colortex8;
uniform sampler2D colortex10;
uniform sampler2D colortex12;
uniform sampler2D colortex13;
uniform sampler2D colortex14;
uniform sampler2D colortex15;

     uniform int renderStage; 
uniform sampler2D colortex0;
uniform sampler2D colortex11;
uniform sampler2D colortex9;
uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D noisetex;
uniform vec2 texelSize;
uniform vec2 viewSize;
uniform float ratio, time;
uniform float viewWidth;
uniform float viewHeight;
uniform float frameTimeCounter;

uniform int frameCounter;
uniform int isEyeInWater;
uniform float rainStrength;
uniform float near;
uniform float aspectRatio;
uniform float far;
#include "/lib/color_transforms.glsl"
#include "/lib/color_dither.glsl"
#include "/lib/noise.glsl"
float cdist(vec2 coord) {
	return max(abs(coord.s-0.5),abs(coord.t-0.5))*2.0;
}

float ld(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}
vec3 closestToCamera3x3()
{
	vec2 du = vec2(texelSize.x, 0.0);
	vec2 dv = vec2(0.0, texelSize.y);

	vec3 dtl = vec3(texcoord,0.) + vec3(-texelSize, texture2D(depthtex0, texcoord - dv - du).x);
	vec3 dtc = vec3(texcoord,0.) + vec3( 0.0, -texelSize.y, texture2D(depthtex0, texcoord - dv).x);
	vec3 dtr = vec3(texcoord,0.) +  vec3( texelSize.x, -texelSize.y, texture2D(depthtex0, texcoord - dv + du).x);

	vec3 dml = vec3(texcoord,0.) +  vec3(-texelSize.x, 0.0, texture2D(depthtex0, texcoord - du).x);
	vec3 dmc = vec3(texcoord,0.) + vec3( 0.0, 0.0, texture2D(depthtex0, texcoord).x);
	vec3 dmr = vec3(texcoord,0.) + vec3( texelSize.x, 0.0, texture2D(depthtex0, texcoord + du).x);

	vec3 dbl = vec3(texcoord,0.) + vec3(-texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv - du).x);
	vec3 dbc = vec3(texcoord,0.) + vec3( 0.0, texelSize.y, texture2D(depthtex0, texcoord + dv).x);
	vec3 dbr = vec3(texcoord,0.) + vec3( texelSize.x, texelSize.y, texture2D(depthtex0, texcoord + dv + du).x);

	vec3 dmin = dmc;

	dmin = dmin.z > dtc.z? dtc : dmin;
	dmin = dmin.z > dtr.z? dtr : dmin;

	dmin = dmin.z > dml.z? dml : dmin;
	dmin = dmin.z > dtl.z? dtl : dmin;
	dmin = dmin.z > dmr.z? dmr : dmin;

	dmin = dmin.z > dbl.z? dbl : dmin;
	dmin = dmin.z > dbc.z? dbc : dmin;
	dmin = dmin.z > dbr.z? dbr : dmin;

	return dmin;
}



vec3 decode3x16(float a){
    int bf = int(a*65535.);
    return vec3(bf%32, (bf>>5)%64, bf>>11) / vec3(31,63,31);
}

float encode2x16(vec2 a){
    ivec2 bf = ivec2(a*255.);
    return float( bf.x|(bf.y<<8) ) / 65535.;
}

vec2 decode2x16(float a){
    int bf = int(a*65535.);
    return vec2(bf%256, bf>>8) / 255.;
}
float encodeNormal3x16(vec3 a){
    vec3 b  = abs(a);
    vec2 p  = a.xy / (b.x + b.y + b.z);
    vec2 sp = vec2(greaterThanEqual(p, vec2(0.0))) * 2.0 - 1.0;

    vec2 encoded = a.z <= 0.0 ? (1.0 - abs(p.yx)) * sp : p;

    encoded = encoded * 0.5 + 0.5;

    return encode2x16(encoded);
}
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
vec3 getDepthPoint(vec2 coord, float depth) {
    vec4 pos;
    pos.xy = coord;
    pos.z = depth;
    pos.w = 1.0;
    pos.xyz = pos.xyz * 2.0 - 1.0; //convert from the 0-1 range to the -1 to +1 range
    pos = gbufferProjectionInverse * pos;
    pos.xyz /= pos.w;
    
    return pos.xyz;
}

vec3 constructNormal(float depthA, vec2 texcoords, sampler2D depthtex) {
    const vec2 offsetB = vec2(0.0,0.001);
    const vec2 offsetC = vec2(0.001,0.0);
  
    float depthB = texture2D(depthtex, texcoords + offsetB).r;
    float depthC = texture2D(depthtex, texcoords + offsetC).r;
  
    vec3 A = getDepthPoint(texcoords, depthA);
	vec3 B = getDepthPoint(texcoords + offsetB, depthB);
	vec3 C = getDepthPoint(texcoords + offsetC, depthC);

	vec3 AB = normalize(B - A);
	vec3 AC = normalize(C - A);

	vec3 normal =  -cross(AB, AC);
	// normal.z = -normal.z;

	return normalize(normal);
}


vec3 decodeNormal3x16(float encoded){
    vec2 a = decode2x16(encoded);

    a = a * 2.0 - 1.0;
    vec2 b = abs(a);
    float z = 1.0 - b.x - b.y;
    vec2 sa = vec2(greaterThanEqual(a, vec2(0.0))) * 2.0 - 1.0;

    vec3 decoded = normalize(vec3(
        z < 0.0 ? (1.0 - b.yx) * sa : a.xy,
        z
    ));

    return decoded;
}

vec3 worldToView(vec3 worldPos) {

    vec4 pos = vec4(worldPos, 0.0);
    pos = gbufferModelView * pos;

    return pos.xyz;
}





vec2 pV[4];
// |0  |1
//
// |2  |3

vec2 pH[3];
//	- 2
//	- 1
//	- 0

vec2 uv;
vec2 pixel;
int SIZE = 8;
vec2 SEGMENT;
float KERNING = 1.3;
const ivec2 DIGITS = ivec2(2, 4);

void globalInit(){
    pixel = 2.0/vec2(viewWidth,viewHeight);
    SEGMENT = pixel * vec2(SIZE, 1.0);
	}

void fillNumbers(){
    pV[0] = vec2(0, SIZE);  pV[1] = vec2(SIZE - 1, SIZE);
    pV[2] = vec2(0, 0); 	pV[3] = vec2(SIZE - 1, 0);
    
    for (int i = 0; i < 3; i++)
    	pH[i] = vec2(0, SIZE * i);
    
	}

vec2 digitSegments(int d){
    vec2 v;
    if (d == 0) v = vec2(.11115, .1015);
    if (d == 1) v = vec2(.01015, .0005);
    if (d == 2) v = vec2(.01105, .1115);
    if (d == 3) v = vec2(.01015, .1115);
    if (d == 4) v = vec2(.11015, .0105);
    if (d == 5) v = vec2(.10015, .1115);
    if (d == 6) v = vec2(.10115, .1115);
    if (d == 7) v = vec2(.01015, .0015);
    if (d == 8) v = vec2(.11115, .1115);
    if (d == 9) v = vec2(.11015, .1115);
    return v;
	}

vec2 step2(vec2 edge, vec2 v){
    return vec2(step(edge.x, v.x), step(edge.y, v.y));
	}

float segmentH(vec2 pos){
    vec2 sv = step2(pos, uv) - step2(pos + SEGMENT.xy, uv);
    return step(1.1, length(sv));
	}

float segmentV(vec2 pos){
    vec2 sv = step2(pos, uv) - step2(pos + SEGMENT.yx, uv);
    return step(1.1, length(sv));
	}

float nextDigit(inout float f){
    f = fract(f) * 10.0;
    return floor(f);
	}

float drawDigit(int d, vec2 pos){
    vec4 sv = vec4(1.0, 0.0, 1.0, 0.0);
    vec3 sh = vec3(1.0);
    float c = 0.0;
    
    vec2 v = digitSegments(d);
    
    for (int i = 0; i < 4; i++)
        c += segmentV(pos + pixel.x * pV[i]) * nextDigit(v.x);

    for (int i = 0; i < 3; i++)
        c += segmentH(pos + pixel.x * pH[i]) * nextDigit(v.y);
    
	return c;
	}

float printNumber(float f, vec2 pos){
    float c = 0.0;
	f += 0.00001;
    f /= pow(10.0, float(DIGITS.x));
        
    for (int i = 0; i < DIGITS.x; i++){
        c += drawDigit(int(nextDigit(f)), pos);
        pos += KERNING * pixel * vec2(SIZE, 0.0);
    	}
    
    for (int i = 0; i < DIGITS.y; i++){
        pos += KERNING * pixel * vec2(SIZE, 0.0);
        c += drawDigit(int(nextDigit(f)), pos);
    	}
   	return c;
	}




vec2 tapLocation(int sampleNumber,int nb, float nbRot,float jitter,float distort)
{
    float alpha = (sampleNumber+jitter)/nb;
    float angle = jitter*6.28+alpha * nbRot * 6.28;
    float sin_v, cos_v;

	sin_v = sin(angle);
	cos_v = cos(angle);

    return vec2(cos_v, sin_v)*alpha;
}



#define s2(a, b)				temp = a; a = min(a, b); b = max(temp, b);
#define mn3(a, b, c)			s2(a, b); s2(a, c);
#define mx3(a, b, c)			s2(b, c); s2(a, c);

#define mnmx3(a, b, c)			mx3(a, b, c); s2(a, b);                                   // 3 exchanges
#define mnmx4(a, b, c, d)		s2(a, b); s2(c, d); s2(a, c); s2(b, d);                   // 4 exchanges
#define mnmx5(a, b, c, d, e)	s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges


#define vec vec3
#define toVec(x) x.rgb
vec3 median2(sampler2D tex1) {

    vec v[9];
    ivec2 ssC = ivec2(gl_FragCoord.xy);
	
	
    // Add the pixels which make up our window to the pixel array.
	
	
    for (int dX = -1; dX <= 1; ++dX) {
        for (int dY = -1; dY <= 1; ++dY) {
            ivec2 offset = ivec2(dX, dY);

            // If a pixel in the window is located at (x+dX, y+dY), put it at index (dX + R)(2R + 1) + (dY + R) of the
            // pixel array. This will fill the pixel array, with the top left pixel of the window at pixel[0] and the
            // bottom right pixel of the window at pixel[N-1].
			
			
            v[(dX + 1) * 3 + (dY + 1)] = toVec(texelFetch(tex1, ssC + offset, 0));
        }
    }

    vec temp;
    // Starting with a subset of size 6, remove the min and max each time
    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
    mnmx5(v[1], v[2], v[3], v[4], v[6]);
    mnmx4(v[2], v[3], v[4], v[7]);
    mnmx3(v[3], v[4], v[8]);
    vec3 result = v[4].rgb;
	
	return result;

}

float checkerboard(in vec2 uv)
{
    vec2 pos = floor(uv);
	float checkerboard = mod(pos.x + mod(pos.y, 2.0), 2.0);
  	return checkerboard;
}
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform sampler2D depthtex2;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferProjection;



vec3 getMotionblur(float depth, bool hand,vec3 color) {
  vec2 texcoord = gl_FragCoord.xy*texelSize;
  		vec4 currentPosition = vec4(texcoord.x * 2.0 - 1.0, texcoord.y * 2.0 - 1.0, 2.0 * texture2D(depthtex2, texcoord.st).x - 1.0, 1.0);

		vec4 fragposition = gbufferProjectionInverse * currentPosition;
		fragposition = gbufferModelViewInverse * fragposition;
		fragposition /= fragposition.w;
		fragposition.xyz += cameraPosition;
		vec4 previousPosition = fragposition;
		previousPosition.xyz -= previousCameraPosition;
		previousPosition = gbufferPreviousModelView * previousPosition;
		previousPosition = gbufferPreviousProjection * previousPosition;
		previousPosition /= previousPosition.w;

		
  	    float uVelocityScale = fps / 20;
		vec2 velocity = (currentPosition - previousPosition).st; 
		vec4 velocity2  =	texture2D(colortex13,texcoord*RENDER_SCALE).rgba;	
	   	   velocity2.r = 	((velocity2.x/ld(depth)*28)*0.2);
	   	   velocity2.g = 	((velocity2.y/ld(depth)*28)*0.6);
	//	if(velocity2.a > 0) velocity = velocity2.xy*10;
		if(hand) velocity *=0.1;
		velocity *=  1.5 * 0.02;
  		const int motionblurSamples      = int(8);
		float dither    = clamp(ditherBluenoiseStatic(),0.9,1);

		      velocity *= uVelocityScale;

		float speed = length(velocity / texelSize);
		int  nSamples = clamp(int(speed), 1, motionblurSamples);

		vec2 coord = texcoord.st ;

 
   for (int i = 1; i < nSamples; ++i) {
      vec2 offset = velocity * (float(i) / float(nSamples - 1) - 0.5);
      color += texture2D(colortex5, coord + offset*dither).rgb;
   }


		color = color / nSamples;
return color;
}

void main() {
/* RENDERTARGETS: 7 */

		float z = ld(texture2D(depthtex0, texcoord.st*RENDER_SCALE).r)*far;


  float vignette = (1.5-dot(texcoord-0.5,texcoord-0.5)*2.);
  
  #ifndef firefly_supression
	vec3 col = texture2D(colortex5,texcoord).rgb;
  #else
	vec3 col =  median2(colortex5);

  #endif
	#ifdef motionblur
		float z2 =  (texture2D(depthtex0, texcoord.st*RENDER_SCALE).r);
		bool hand = z < 0.54;
 		col =  getMotionblur(z2,  hand,col) ;
	#endif
  
  
	float noise = blueNoise()*6.28318530718;
	float pcoc = 0;

    fillNumbers();
    uv = gl_FragCoord.xy / vec2(viewWidth,viewHeight).xy;
	#ifdef DOF


		/*--------------------------------*/

		#if DOF_MODE == 0
			float focus = rodExposureDepth.y*far;
		#endif	
		#if DOF_MODE == 1
			float focus = MANUAL_FOCUS;
		#endif			
		
		#if DOF_MODE == 2
				focus = MANUAL_FOCUS * screenBrightness;
		#endif
			for ( int i = 0; i < 15; i++) {
				pcoc += texture2D(colortex11, texcoord.xy + poisson15[i]*0.01).r;
			}
			pcoc = pcoc/15.0;		

		pcoc = (min(abs((focal / aperture) * (focal/1000.0 * (z - focus)) / (z * (focus - focal/1000.0))),texelSize.x*15.0));


		

		 
			#if EXCLUDE_MODE == 2
						pcoc *= float(z > 0.56);
			#endif			
			#if EXCLUDE_MODE == 1
				pcoc *= float(z > focus);
			#endif
	

		mat2 noiseM = mat2( cos( noise ), -sin( noise ),
	                       sin( noise ), cos( noise )
	                         );
		vec3 bcolor = vec3(0.);
		float nb = 0.0;
		vec2 bcoord = vec2(0.0);



		/*--------------------------------*/
			#if BOKEH_MODE == 0 // Standard Bokeh
				bcolor = col;
				for ( int i = 0; i < 60; i++) {
					bcolor += texture2D(colortex5, texcoord.xy + dof_offsets[i]*pcoc*vec2(1.0,aspectRatio)).rgb;
				}
				col = bcolor/61.0;
			#endif
			#if BOKEH_MODE == 1 // Hexagonal Bokeh
				bcolor = col;
				for ( int i = 0; i < 60; i++) {
					bcolor += texture2D(colortex5, texcoord.xy + hex_offsets[i]*pcoc*vec2(1.0,aspectRatio)).rgb;
				}
				col = bcolor/61.0;
			#endif	
			#if BOKEH_MODE == 2 // HQ Bokeh
				for ( int i = 0; i < 209; i++) {
					bcolor += texture2D(colortex5, texcoord.xy + noiseM*shadow_offsets[i]*pcoc*vec2(1.0,aspectRatio)).rgb;
					
				}
				col = bcolor/209.0;
			#endif
			#if BOKEH_MODE == 3 // Paint/Brush Bokeh
				bcolor = col;
				vec3 t = vec3 (0.0);
				float h = 1.004;
				vec2 d = vec2(pcoc / aspectRatio, pcoc);
				for (int i = 0; i < 6; ++i) {
					bcolor = texture2D(colortex5, texcoord.xy + paint_offsets[i]*d).rgb;
					 t = max(sign(bcolor - col), 0.0);
				
				col += (bcolor - col) * t;
				}
				bcolor = col;

			
			#endif
#endif
	vec2 clampedRes = max(vec2(viewWidth,viewHeight),vec2(1920.0,1080.));

	vec3 bloom = texture2D(colortex3,texcoord/clampedRes*vec2(1920.,1080.)*0.5*BLOOM_QUALITY).rgb*0.5*0.14;

	float lightScat = clamp(BLOOM_STRENGTH*0.05*pow(exposure.a,0.2),0.0,1.0)*vignette;

  float VL_abs = texture2D(colortex7,texcoord*RENDER_SCALE).r;
	float purkinje = rodExposureDepth.x/(1.0+rodExposureDepth.x)*Purkinje_strength;
  VL_abs = clamp((1.0-VL_abs)*BLOOMY_FOG*0.75*(1.0-purkinje*0.3)*(1.0+rainStrength),0.0,1.0)*clamp(1.0-pow(cdist(texcoord.xy),15.0),0.0,1.0);
	col = (mix(col,bloom,VL_abs)+bloom*lightScat)*exposure.rgb;

	//Purkinje Effect
  float lum = dot(col,vec3(0.15,0.3,0.55));
	float lum2 = dot(col,vec3(0.85,0.7,0.45))/2;
	float rodLum = lum2*400.;
	
	float rodCurve = mix(1.0, rodLum/(2.5+rodLum), purkinje);
	col = mix(clamp(lum,0.0,0.05)*Purkinje_Multiplier*vec3(Purkinje_R, Purkinje_G, Purkinje_B)+1.5e-3, col, rodCurve);
//	col =vec3(rodCurve);
//	if (col.r > 0.85*3.0) col = vec3(100,0.0,0.0);
//   col = vec3(texture2D(colortex15,texcoord*RENDER_SCALE).rg,0);
	#ifndef USE_ACES_COLORSPACE_APPROXIMATION
  	col = LinearTosRGB(TONEMAP(col));

	#else
		col = col * ACESInputMat;
		col = TONEMAP(col);
		col = LinearTosRGB(clamp(col * ACESOutputMat, 0.0, 1.0));
	#endif
//	col = ACESFitted(texture2D(colortex4,texcoord/3.).rgb/500.);
	gl_FragData[0].rgb = clamp(int8Dither(col,texcoord),0.0,1.0);
	





//  gl_FragData[0].rgb = vec3(  getMotionblur(z2,  hand) 	);

//	globalInit();
//  gl_FragData[0].rgb += vec3(printNumber( fps, vec2(0.48,0.47)));	
//  gl_FragData[0].rgb += vec3(printNumber((RENDER_SCALE_X), vec2(0.48,0.52)));
//  gl_FragData[0].rgb = constructNormal(texture2D(depthtex0, texcoord.st*RENDER_SCALE).r, texcoord*RENDER_SCALE, depthtex0);
//	if (nightMode < 0.99 && texcoord.x < 0.5)	gl_FragData[0].rgb =vec3(0.0,1.0,0.0);

}
