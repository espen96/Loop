

vec3 screenToViewSpace(vec3 screenpos, mat4 projInv, const bool taaAware)
{
    screenpos = screenpos * 2.0 - 1.0;

    vec3 viewpos = vec3(vec2(projInv[0].x, projInv[1].y) * screenpos.xy + projInv[3].xy, projInv[3].z);
    viewpos /= projInv[2].w * screenpos.z + projInv[3].w;

    return viewpos;
}
vec3 screenToViewSpace(vec3 screenpos, mat4 projInv)
{
    return screenToViewSpace(screenpos, projInv, true);
}

vec3 screenToViewSpace(vec3 screenpos)
{
    return screenToViewSpace(screenpos, gbufferProjectionInverse);
}
vec3 screenToViewSpace(vec3 screenpos, const bool taaAware)
{
    return screenToViewSpace(screenpos, gbufferProjectionInverse, taaAware);
}
float sqr(float x)
{
    return x * x;
}

const float rlog2 = 1.0 / log(2.0);
#define expf(x) exp2((x)*rlog2)

float computeVariance(sampler2D tex, ivec2 pos)
{
    float sum_msqr = 0.0;
    float sum_mean = 0.0;

    for (int i = 0; i < 9; i++)
    {
        ivec2 deltaPos = kernelO_3x3[i];

        vec3 col = texelFetch2D(tex, pos + deltaPos, 0).rgb;
        float lum = luma(col);

        sum_msqr += sqr(lum);
        sum_mean += lum;
    }
    sum_msqr /= 9.0;
    sum_mean /= 9.0;

    return abs(sum_msqr - sqr(sum_mean)) * 1 / (max(sum_mean, 1e-25));
}

#ifdef power

float Pow2(float x)
{
    return x * x;
}
vec2 Pow2(vec2 x)
{
    return x * x;
}
vec3 Pow2(vec3 x)
{
    return x * x;
}
vec4 Pow2(vec4 x)
{
    return x * x;
}
float Pow3(float x)
{
    return x * x * x;
}
float Pow4(float x)
{
    x *= x;
    return x * x;
}
vec2 Pow4(vec2 x)
{
    x *= x;
    return x * x;
}
vec3 Pow4(vec3 x)
{
    x *= x;
    return x * x;
}
float Pow5(float x)
{
    float x2 = x * x;
    return x2 * x2 * x;
}
float Pow6(float x)
{
    x *= x;
    return x * x * x;
}
float Pow8(float x)
{
    x *= x;
    x *= x;
    return x * x;
}
float Pow12(float x)
{
    x *= x;
    x *= x;
    return x * x * x;
}
float Pow16(float x)
{
    x *= x;
    x *= x;
    x *= x;
    return x * x;
}

#endif
#ifdef denoise

uniform sampler2D colortex4;

#endif

vec3 atrous3(vec2 coord, const int size, sampler2D tex1, float extraweight)
{

    ivec2 pos = ivec2(floor(coord * vec2(viewWidth, viewHeight) / RENDER_SCALE));
    vec3 colorCenter = texelFetch(tex1, pos, 0).rgb;
    float variance = computeVariance(colortex8, ivec2(floor(coord * vec2(viewWidth, viewHeight) / RENDER_SCALE)));
    vec2 moment = texelFetch(colortex15, pos, 0).rg;
    if (variance == 0)
        return colorCenter;
    float weight = 0.0;
    vec4 normaldepth = texelFetch(colortex10, pos, 0).rgba;
    float c_depth = normaldepth.a * far;
    vec3 origNormal = normaldepth.rgb;

    vec3 totalColor = colorCenter;

    float totalWeight = 1.0;

    float var2 = 1 / (0.5 + variance * 4);

    //#define HQ

    const int r = 1;

#ifdef HQ
    for (int i = 0; i < 25; i++)
    {
        ivec2 delta = kernelO_5x5[i] * size;
#else
    for (int i = 0; i < 9; i++)
    {
        ivec2 delta = kernelO_3x3[i] * size;
#endif

        if (delta.x == 0 && delta.y == 0)
            continue;

        ivec2 d_pos = pos + delta;

        vec4 normaldepth2 = texelFetch(colortex10, d_pos, 0).rgba;
        float cu_depth = (normaldepth2.a) * far;

        vec3 normal = (normaldepth2.rgb);

        vec3 color = texelFetch(tex1, d_pos, 0).rgb;
        if (distance(luma(color), luma(colorCenter)) < 0.0)
            return totalColor;
        float variance2 = computeVariance(colortex8, d_pos);

        float d_weight = abs(cu_depth - c_depth);
        float depthWeight = expf(-d_weight);
        if (depthWeight < 1e-5)
            continue;

        float normalWeight = pow(clamp(dot(normal, origNormal), 0, 1), 32);

        float weight = normalWeight;

        weight *= exp(-d_weight - var2);

        totalColor += color.rgb * weight;

        totalWeight += weight;
    }

    totalColor *= 1 / (max(totalWeight, 1e-25));

    return totalColor;
}
