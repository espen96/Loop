
//Vignetting, applies bloom, applies exposure and tonemaps the final image
#extension GL_EXT_gpu_shader4 : enable
//#define BICUBIC_UPSCALING //Provides a better interpolation when using a render quality different of 1.0, slower
#define CONTRAST_ADAPTATIVE_SHARPENING
#define SHARPENING 0.5 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]
#define SATURATION 0.14 // Negative values desaturates colors, Positive values saturates color, 0 is no change [-1.0 -0.98 -0.96 -0.94 -0.92 -0.9 -0.88 -0.86 -0.84 -0.82 -0.8 -0.78 -0.76 -0.74 -0.72 -0.7 -0.68 -0.66 -0.64 -0.62 -0.6 -0.58 -0.56 -0.54 -0.52 -0.5 -0.48 -0.46 -0.44 -0.42 -0.4 -0.38 -0.36 -0.34 -0.32 -0.3 -0.28 -0.26 -0.24 -0.22 -0.2 -0.18 -0.16 -0.14 -0.12 -0.1 -0.08 -0.06 -0.04 -0.02 0.0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 1.0 ]
#define CROSSTALK 0.25 // Desaturates bright colors and preserves saturation in darker areas (inverted if negative). Helps avoiding almsost fluorescent colors [-1.0 -0.98 -0.96 -0.94 -0.92 -0.9 -0.88 -0.86 -0.84 -0.82 -0.8 -0.78 -0.76 -0.74 -0.72 -0.7 -0.68 -0.66 -0.64 -0.62 -0.6 -0.58 -0.56 -0.54 -0.52 -0.5 -0.48 -0.46 -0.44 -0.42 -0.4 -0.38 -0.36 -0.34 -0.32 -0.3 -0.28 -0.26 -0.24 -0.22 -0.2 -0.18 -0.16 -0.14 -0.12 -0.1 -0.08 -0.06 -0.04 -0.02 0.0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 1.0 ]
in vec2 texcoord;

uniform sampler2D colortex7;
uniform vec2 texelSize;
uniform vec2 viewSize;
uniform float viewWidth;
uniform float viewHeight;
uniform float frameTimeCounter;
uniform int frameCounter;
uniform int isEyeInWater;
uniform int hideGUI;
#include "/lib/color_transforms.glsl"
#include "/lib/color_dither.glsl"
#include "/lib/res_params.glsl"
#include "/lib/Shadow_Params.glsl"

#define DEBUG
#define DEBUG_PROGRAM 50 // [-10 -1 30 31 32 50]
#define DEBUG_BRIGHTNESS 1.0 // [1/65536.0 1/32768.0 1/16384.0 1/8192.0 1/4096.0 1/2048.0 1/1024.0 1/512.0 1/256.0 1/128.0 1/64.0 1/32.0 1/16.0 1/8.0 1/4.0 1/2.0 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0 512.0 1024.0 2048.0 4096.0 8192.0 16384.0 32768.0 65536.0]
#define DRAW_DEBUG_VALUE

vec3 Debug = vec3(0.0);

// Write the direct variable onto the screen
void show(bool x) {
	Debug = vec3(float(x));
}
void show(float x) {
	Debug = vec3(x);
}
void show(vec2 x) {
	Debug = vec3(x, 0.0);
}
void show(vec3 x) {
	Debug = x;
}
void show(vec4 x) {
	Debug = x.rgb;
}

void inc(bool x) {
	Debug += vec3(float(x));
}
void inc(float x) {
	Debug += vec3(x);
}
void inc(vec2 x) {
	Debug += vec3(x, 0.0);
}
void inc(vec3 x) {
	Debug += x;
}
void inc(vec4 x) {
	Debug += x.rgb;
}

#ifdef DRAW_DEBUG_VALUE
	// Display the value of the variable on the debug value viewer
	#define showval(x) if (all(equal(ivec2(gl_FragCoord.xy), ivec2(viewSize/2)))) show(x);
	#define incval(x)  if (all(equal(ivec2(gl_FragCoord.xy), ivec2(viewSize/2)))) inc(x);
#else
	#define showval(x)
	#define incval(x)
#endif

vec4 SampleTextureCatmullRom(sampler2D tex, vec2 uv, vec2 texSize) {
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
	vec2 samplePos = uv * texSize;
	vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
	vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
	vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
	vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
	vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
	vec2 w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
	vec2 w12 = w1 + w2;
	vec2 offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
	vec2 texPos0 = texPos1 - vec2(1.0);
	vec2 texPos3 = texPos1 + vec2(2.0);
	vec2 texPos12 = texPos1 + offset12;

	texPos0 *= texelSize;
	texPos3 *= texelSize;
	texPos12 *= texelSize;

	vec4 result = vec4(0.0);
	result += texture(tex, vec2(texPos0.x, texPos0.y)) * w0.x * w0.y;
	result += texture(tex, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
	result += texture(tex, vec2(texPos3.x, texPos0.y)) * w3.x * w0.y;

	result += texture(tex, vec2(texPos0.x, texPos12.y)) * w0.x * w12.y;
	result += texture(tex, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
	result += texture(tex, vec2(texPos3.x, texPos12.y)) * w3.x * w12.y;

	result += texture(tex, vec2(texPos0.x, texPos3.y)) * w0.x * w3.y;
	result += texture(tex, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
	result += texture(tex, vec2(texPos3.x, texPos3.y)) * w3.x * w3.y;

	return result;
}
/***********************************************************************/
/* Text Rendering */
const int _A = 0x64bd29, _B = 0x749d27, _C = 0xe0842e, _D = 0x74a527, _E = 0xf09c2f, _F = 0xf09c21, _G = 0xe0b526, _H = 0x94bd29, _I = 0xf2108f, _J = 0x842526, _K = 0x9284a9, _L = 0x10842f, _M = 0x97a529, _N = 0x95b529, _O = 0x64a526, _P = 0x74a4e1, _Q = 0x64acaa, _R = 0x749ca9, _S = 0xe09907, _T = 0xf21084, _U = 0x94a526, _V = 0x94a544, _W = 0x94a5e9, _X = 0x949929, _Y = 0x94b90e, _Z = 0xf4106f, _0 = 0x65b526, _1 = 0x431084, _2 = 0x64904f, _3 = 0x649126, _4 = 0x94bd08, _5 = 0xf09907, _6 = 0x609d26, _7 = 0xf41041, _8 = 0x649926, _9 = 0x64b904, _APST = 0x631000, _PI = 0x07a949, _UNDS = 0x00000f, _HYPH = 0x001800, _TILD = 0x051400, _PLUS = 0x011c40, _EQUL = 0x0781e0, _SLSH = 0x041041, _EXCL = 0x318c03, _QUES = 0x649004, _COMM = 0x000062, _FSTP = 0x000002, _QUOT = 0x528000, _BLNK = 0x000000, _COLN = 0x000802, _LPAR = 0x410844, _RPAR = 0x221082;

const ivec2 MAP_SIZE = ivec2(5, 5);

int GetBit(int bitMap, int index) {
	return (bitMap >> index) & 1;
}

float DrawChar(int charBitMap, inout vec2 anchor, vec2 charSize, vec2 uv) {
	uv = (uv - anchor) / charSize;

	anchor.x += charSize.x;

	if (!all(lessThan(abs(uv - vec2(0.5)), vec2(0.5))))
		return 0.0;

	uv *= MAP_SIZE;

	int index = int(uv.x) % MAP_SIZE.x + int(uv.y) * MAP_SIZE.x;

	return float(GetBit(charBitMap, index));
}

const int STRING_LENGTH = 15;
int[STRING_LENGTH] drawString;

float DrawString(inout vec2 anchor, vec2 charSize, int stringLength, vec2 uv) {
	uv = (uv - anchor) / charSize;

	anchor.x += charSize.x * stringLength;

	if (!all(lessThan(abs(uv / vec2(stringLength, 1.0) - vec2(0.5)), vec2(0.5))))
		return 0.0;

	int charBitMap = drawString[int(uv.x)];

	uv *= MAP_SIZE;

	int index = int(uv.x) % MAP_SIZE.x + int(uv.y) * MAP_SIZE.x;

	return float(GetBit(charBitMap, index));
}

#define log10(x) (log2(x) / log2(10.0))

float DrawInt(int val, inout vec2 anchor, vec2 charSize, vec2 uv) {
	if (val == 0)
		return DrawChar(_0, anchor, charSize, uv);

	const int _DIGITS[10] = int[10](_0, _1, _2, _3, _4, _5, _6, _7, _8, _9);

	bool isNegative = val < 0.0;

	if (isNegative)
		drawString[0] = _HYPH;

	val = abs(val);

	int posPlaces = int(ceil(log10(abs(val) + 0.001)));
	int strIndex = posPlaces - int(!isNegative);

	while (val > 0) {
		drawString[strIndex--] = _DIGITS[val % 10];
		val /= 10;
	}

	return DrawString(anchor, charSize, posPlaces + int(isNegative), texcoord);
}

float DrawFloat(float val, inout vec2 anchor, vec2 charSize, int negPlaces, vec2 uv) {
	int whole = int(val);
	int part = int(fract(abs(val)) * pow(10, negPlaces));

	int posPlaces = max(int(ceil(log10(abs(val)))), 1);

	anchor.x -= charSize.x * (posPlaces + int(val < 0) + 0.25);
	float ret = 0.0;
	ret += DrawInt(whole, anchor, charSize, uv);
	ret += DrawChar(_FSTP, anchor, charSize, texcoord);
	anchor.x -= charSize.x * 0.3;
	ret += DrawInt(part, anchor, charSize, uv);

	return ret;
}

void DrawDebugText() {
	#if (defined DEBUG) && (defined DRAW_DEBUG_VALUE) && (DEBUG_PROGRAM != 50)
	vec2 charSize = vec2(0.009) * viewSize.yy / viewSize;
	vec2 texPos = vec2(charSize.x / 1.5, 1.0 - charSize.y * 2.0);

	if (hideGUI != 0 || texcoord.x > charSize.x * 12.0 || texcoord.y < 1 - charSize.y * 12) {
		return;
	}

	vec3 color = vec3(gl_FragColor.rgb) * 1;
	float text = 0.0;

	vec3 val = texelFetch(colortex7, ivec2(viewSize / 2.0), 0).rgb;

	drawString = int[STRING_LENGTH](_D, _E, _B, _U, _G, 0, _S, _T, _A, _T, _S, 0, 0, 0, 0);
	text += DrawString(texPos, charSize, 11, texcoord);
	color += text * vec3(1.0, 1.0, 1.0) * sqrt(clamp(abs(val.b), 0.2, 1.0));

	texPos.x = charSize.x / 1.0, 1.0;
	texPos.y -= charSize.y * 2;

	text = 0.0;
	drawString = int[STRING_LENGTH](_R, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	text += DrawString(texPos, charSize, 2, texcoord);
	texPos.x += charSize.x * 5.0;
	text += DrawFloat(val.r, texPos, charSize, 3, texcoord);
	color += text * vec3(1.0, 0.0, 0.0) * sqrt(clamp(abs(val.r), 0.2, 1.0));

	texPos.x = charSize.x / 1.0, 1.0;
	texPos.y -= charSize.y * 1.4;

	text = 0.0;
	drawString = int[STRING_LENGTH](_G, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	text += DrawString(texPos, charSize, 2, texcoord);
	texPos.x += charSize.x * 5.0;
	text += DrawFloat(val.g, texPos, charSize, 3, texcoord);
	color += text * vec3(0.0, 1.0, 0.0) * sqrt(clamp(abs(val.g), 0.2, 1.0));

	texPos.x = charSize.x / 1.0, 1.0;
	texPos.y -= charSize.y * 1.4;

	text = 0.0;
	drawString = int[STRING_LENGTH](_B, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	text += DrawString(texPos, charSize, 2, texcoord);
	texPos.x += charSize.x * 5.0;
	text += DrawFloat(val.b, texPos, charSize, 3, texcoord);
	color += text * vec3(0.0, 0.8, 1.0) * sqrt(clamp(abs(val.b), 0.2, 1.0));

	texPos.x = charSize.x / 1.0, 1.0;
	texPos.y -= charSize.y * 1.4;

	text = 0.0;
	drawString = int[STRING_LENGTH](_F, _P, _S, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	text += DrawString(texPos, charSize, 4, texcoord);
	texPos.x += charSize.x * 4.0;
	text += DrawFloat(fps, texPos, charSize, 2, texcoord);
	color += text * vec3(1.0, 1.0, 1.0) * sqrt(clamp(abs(val.b), 0.2, 1.0));

	texPos.x = charSize.x / 1.0, 1.0;
	texPos.y -= charSize.y * 1.4;

	text = 0.0;
	drawString = int[STRING_LENGTH](_R, _E, _S, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	text += DrawString(texPos, charSize, 4, texcoord);
	texPos.x += charSize.x * 3.0;
	text += DrawFloat(RENDER_SCALE_X, texPos, charSize, 3, texcoord);
	color += text * vec3(1.0, 1.0, 1.0) * sqrt(clamp(abs(val.b), 0.2, 1.0));

	texPos.x = charSize.x / 1.0, 1.0;
	texPos.y -= charSize.y * 1.4;

	text = 0.0;
	drawString = int[STRING_LENGTH](_R, _A, _Y, _COLN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	text += DrawString(texPos, charSize, 4, texcoord);
	texPos.x += charSize.x * 3.0;
	text += DrawFloat(RAY_COUNT, texPos, charSize, 3, texcoord);
	color += text * vec3(1.0, 1.0, 1.0) * sqrt(clamp(abs(val.b), 0.2, 1.0));

	gl_FragColor.rgb = color;
	#endif
}
/***********************************************************************/
void main() {
  #ifdef BICUBIC_UPSCALING
	vec3 col = SampleTextureCatmullRom(colortex7, texcoord, 1.0 / texelSize).rgb;
  #else
	vec3 col = texture(colortex7, texcoord).rgb;
  #endif

  #ifdef CONTRAST_ADAPTATIVE_SHARPENING
    //Weights : 1 in the center, 0.5 middle, 0.25 corners
	vec3 albedoCurrent1 = texture(colortex7, texcoord + vec2(texelSize.x, texelSize.y) / MC_RENDER_QUALITY * 0.5).rgb;
	vec3 albedoCurrent2 = texture(colortex7, texcoord + vec2(texelSize.x, -texelSize.y) / MC_RENDER_QUALITY * 0.5).rgb;
	vec3 albedoCurrent3 = texture(colortex7, texcoord + vec2(-texelSize.x, -texelSize.y) / MC_RENDER_QUALITY * 0.5).rgb;
	vec3 albedoCurrent4 = texture(colortex7, texcoord + vec2(-texelSize.x, texelSize.y) / MC_RENDER_QUALITY * 0.5).rgb;

	vec3 m1 = -0.5 / 3.5 * col + albedoCurrent1 / 3.5 + albedoCurrent2 / 3.5 + albedoCurrent3 / 3.5 + albedoCurrent4 / 3.5;
	vec3 std = abs(col - m1) + abs(albedoCurrent1 - m1) + abs(albedoCurrent2 - m1) +
		abs(albedoCurrent3 - m1) + abs(albedoCurrent3 - m1) + abs(albedoCurrent4 - m1);
	float contrast = 1.0 - luma(std) / 5.0;
	col = col * (1.0 + (SHARPENING + UPSCALING_SHARPNENING) * contrast) - (SHARPENING + UPSCALING_SHARPNENING) / (1.0 - 0.5 / 3.5) * contrast * (m1 - 0.5 / 3.5 * col);
  #endif

	float lum = luma(col);
	vec3 diff = col - lum;
	col = col + diff * (-lum * CROSSTALK + SATURATION);
  //col = -vec3(-lum*CROSSFADING + SATURATION);
	gl_FragColor.rgb = clamp(int8Dither(col, texcoord), 0.0, 1.0);
  //gl_FragColor.rgb = vec3(contrast);
  //gl_FragColor.rgb = vec3(contrast);
	DrawDebugText();
}
