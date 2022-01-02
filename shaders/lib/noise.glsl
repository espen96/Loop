

float interleaved_gradientNoise(float temporal)
{
    vec2 coord = gl_FragCoord.xy;
    float noise = fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y) + temporal);
    return noise;
}
float blueNoise()
{
    return fract(texelFetch2D(noisetex, ivec2(gl_FragCoord.xy) % 512, 0).a + 1.0 / 1.6180339887 * frameCounter);
}

#ifndef gbuffer

float blueNoiseFloat(vec2 coord)
{
    return texelFetch2D(noisetex, ivec2(gl_FragCoord.xy) % 512, 0).a;
}

vec4 blueNoise(vec2 coord)
{
    return texelFetch2D(colortex2, ivec2(coord) % 512, 0);
}
float R2_dither()
{
    vec2 alpha = vec2(0.75487765, 0.56984026);
    return fract(alpha.x * gl_FragCoord.x + alpha.y * gl_FragCoord.y + 1.0 / 1.6180339887 * frameCounter);
}

float ditherBluenoiseStatic()
{
    ivec2 coord = ivec2(fract(gl_FragCoord.xy / 256.0) * 256.0);
    float noise = texelFetch2D(noisetex, coord, 0).a;

    return noise;
}

float interleaved_gradientNoise()
{
    vec2 coord = gl_FragCoord.xy;
    float noise = fract(52.9829189 * fract(0.06711056 * coord.x + 0.00583715 * coord.y) + frameCounter / 1.6180339887);
    return noise;
}
#else

float interleaved_gradientNoise()
{
    return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y) +
                 frameTimeCounter * 51.9521);
}

#endif