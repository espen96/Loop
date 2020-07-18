#version 120
//Vignetting, applies bloom, applies exposure and tonemaps the final image
#extension GL_EXT_gpu_shader4 : enable

#include "/lib/settings.glsl"

varying vec2 texcoord;

uniform sampler2D colortex7;

uniform vec2 texelSize;

uniform float rainStrength;
uniform float viewWidth;
uniform float viewHeight;
uniform float frameTimeCounter;
uniform float sunAngle;					   
uniform int frameCounter;
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform int isEyeInWater;
#include "/lib/color_transforms.glsl"
#include "/lib/color_dither.glsl"
#include "/lib/encode.glsl"

		vec4 data = texture2D(colortex1,texcoord);
		vec4 dataUnpacked0 = vec4(decodeVec2(data.x),decodeVec2(data.y));
		vec4 dataUnpacked1 = vec4(decodeVec2(data.z),decodeVec2(data.w));

		vec3 albedo = toLinear(vec3(dataUnpacked0.xz,dataUnpacked1.x));
		bool emissive = abs(dataUnpacked1.w-0.9) <0.01;


vec4 SampleTextureCatmullRom(sampler2D tex, vec2 uv, vec2 texSize )
{
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
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
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
    result += texture2D(tex, vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += texture2D(tex, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += texture2D(tex, vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += texture2D(tex, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += texture2D(tex, vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += texture2D(tex, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture2D(tex, vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
}




void main() {
  #ifdef BICUBIC_UPSCALING
    vec3 col = SampleTextureCatmullRom(colortex7,texcoord,1.0/texelSize).rgb;
  #else
    vec3 col = texture2D(colortex7,texcoord).rgb;
  #endif

  #ifdef CONTRAST_ADAPTATIVE_SHARPENING
    vec3 albedoCurrent1 = texture2D(colortex7, texcoord + vec2(texelSize.x,texelSize.y)).rgb;
    vec3 albedoCurrent2 = texture2D(colortex7, texcoord + vec2(texelSize.x,-texelSize.y)).rgb;
    vec3 albedoCurrent3 = texture2D(colortex7, texcoord + vec2(-texelSize.x,-texelSize.y)).rgb;
    vec3 albedoCurrent4 = texture2D(colortex7, texcoord + vec2(-texelSize.x,texelSize.y)).rgb;
    vec3 albedoCurrent5 = texture2D(colortex7, texcoord + vec2(0.0,texelSize.y)).rgb;
    vec3 albedoCurrent6 = texture2D(colortex7, texcoord + vec2(0.0,-texelSize.y)).rgb;
    vec3 albedoCurrent7 = texture2D(colortex7, texcoord + vec2(-texelSize.x,0.0)).rgb;
    vec3 albedoCurrent8 = texture2D(colortex7, texcoord + vec2(texelSize.x,0.0)).rgb;

    vec3 m1 = (col + albedoCurrent1 + albedoCurrent2 + albedoCurrent3 + albedoCurrent4 + albedoCurrent5 + albedoCurrent6 + albedoCurrent7 + albedoCurrent8)/9.0;
    vec3 std = abs(col - m1) + abs(albedoCurrent1 - m1) + abs(albedoCurrent2 - m1) +
     abs(albedoCurrent3 - m1) + abs(albedoCurrent3 - m1) + abs(albedoCurrent4 - m1) +
     abs(albedoCurrent5 - m1) + abs(albedoCurrent6 - m1) + abs(albedoCurrent7 - m1) +
     abs(albedoCurrent8 - m1);
    float contrast = 1.0 - luma(std)/9.0;
    col = col*(1.0+SHARPENING*contrast)
          - (albedoCurrent5 + albedoCurrent6 + albedoCurrent7 + albedoCurrent8 + (albedoCurrent1 + albedoCurrent2 + albedoCurrent3 + albedoCurrent4)/2.0)/6.0 * SHARPENING*contrast;
  #endif
  vec3 emc = texture2D(colortex7,texcoord).rgb;
  float modWT = sunAngle*10;
  float modWT2 = -smoothstep(5,5.5,modWT);  
  float modWT3 = max(modWT2, -0.2);							
  float lum = luma(col);
  vec3 diff = col-lum;
  col = col + diff*(-lum*(CROSSTALK+modWT3) + SATURATION-(rainStrength/4));


	gl_FragColor.rgb = clamp(int8Dither(col,texcoord),0.0,1.0);

}
