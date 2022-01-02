float bayer2(vec2 a)
{
    a = floor(a);
    return fract(dot(a, vec2(.5, a.y * .75)));
}

#define bayer4(a) (bayer2(.5 * (a)) * .25 + bayer2(a))
#define bayer8(a) (bayer4(.5 * (a)) * .25 + bayer2(a))
#define bayer16(a) (bayer8(.5 * (a)) * .25 + bayer2(a))
#define bayer32(a) (bayer16(.5 * (a)) * .25 + bayer2(a))
#define bayer64(a) (bayer32(.5 * (a)) * .25 + bayer2(a))

#define fLengthSource(x) sqrt(dotX(x))
#define dotXSource(x) dot(x, x)

float RGBValue = 0.003921568627451;

vec2 sincos(float x)
{
    return vec2(sin(x), cos(x));
}

float dotX(vec4 x)
{
    return dotXSource(x);
}

float fLength(vec4 x)
{
    return fLengthSource(x);
}

const float PI = 3.14159265359;
const float goldenAngle = 137.5077640500378546463487;

vec2 circlemap(float i, float n)
{
    return sincos(i * n * goldenAngle) * sqrt(i);
}

vec3 getRSM(vec3 normal, bool entity, vec3 albedo, vec2 lightmap, float z)
{
    vec2 texcoord = gl_FragCoord.xy * texelSize;
    if (lightmap.y <= 0.0)
        return vec3(0.0);
    //	float noise = bayer64(gl_FragCoord.xy);
    //		  noise = fract(frameCounter * (1.0 / 9.0) + noise);
    float noise = blueNoise();
    float Depth = texture(depthtex1, texcoord).x;
    vec4 fragpos =
        gbufferProjectionInverse *
        (vec4((texcoord / RENDER_SCALE) - vec2(TAA_Offset * vec2(texelSize.x, texelSize.y)) * 0.5, Depth, 1.0) * 2.0 -
         1.0);
    fragpos /= fragpos.w;
    float blockDistance = fLength(fragpos);
    float giDistanceMask = clamp(1.0 - (blockDistance * 0.0075), 0.0, 1.0);
    if (giDistanceMask <= 0.0)
        return vec3(0.0);

    vec4 worldposition = gbufferModelViewInverse * vec4(fragpos);
    vec3 wposr = worldposition.xyz;
    worldposition = shadowModelView * worldposition;
    vec3 wpos2 = worldposition.xyz;
    worldposition = shadowProjection * worldposition;
    worldposition /= worldposition.w;

    worldposition = worldposition * 0.5 + 0.5;

    vec3 t1 = mat3(shadowModelView) * normal;

    vec3 projNormal = t1.rgb;
    const int steps = 2;
    const float rSteps = 1.0 / steps;
    vec3 gi = vec3(0.0);
    const float offsetSize = 100;
    const float rOffsetSize = 1.0 / offsetSize;

    //	for (int i = 0; i < steps; ++i){
    for (float i = 1.0; i < 2.0; i += steps)
    { // Faster but needs some extra denoising

        vec2 offset = circlemap((float(i) + noise) * rSteps, 4096.0 * float(steps)) * offsetSize * (1.0 / 2048.0);
        float weight = length(offset);
        if (weight <= 0.01 || weight >= 0.1)
            break;

        vec3 p3 = mat3(gbufferModelViewInverse) * fragpos.xyz;
        p3 += gbufferModelViewInverse[3].xyz;

        vec2 TC = ((worldposition.xy + offset * 0.95) * 2.0 - 1.0) * 0.5 + 0.5;
        if (TC.x <= 0.25 || TC.y <= 0.25)
            break;
        vec3 GIprojectedShadowPosition = mat3(shadowModelView) * p3 + shadowModelView[3].xyz;
        GIprojectedShadowPosition = diagonal3(shadowProjection) * GIprojectedShadowPosition + shadowProjection[3].xyz;

        // apply distortion
        float GIdistortFactor = calcDistort(GIprojectedShadowPosition.xy);
        GIprojectedShadowPosition.xy *= GIdistortFactor;

        vec3 GIprojectedShadowPosition2 = GIprojectedShadowPosition * vec3(0.5, 0.5, 0.5) + vec3(0.5, 0.5, 0.5);
        vec2 GIprojectedShadowPosition3 =
            ((GIprojectedShadowPosition2.xy * shadowMapResolution) + (offset * shadowMapResolution));

        vec4 shadowSample = vec4(texelFetch2D(shadowcolor0, ivec2(GIprojectedShadowPosition3.xy), 0).rgb,
                                 texelFetch2D(shadow, ivec2(GIprojectedShadowPosition3), 0).x);

        vec3 samplePostion = vec3(offset.xy, shadowSample.a * 8.0 - 4.0) - GIprojectedShadowPosition2;
        float normFactor = dot(samplePostion, samplePostion);

        float falloff = 1.0 / (normFactor * rOffsetSize * 16384.0 + rOffsetSize * 16.0);
        if (falloff <= 0.0045)
            break;

        vec3 sAlbedo = (shadowSample.rgb);
        float salbl = length(sAlbedo);
        float albl = length(albedo);

        if (salbl / albl >= 0.98 && salbl / albl <= 1.02)
            break;

        float sampleDepth = shadowSample.a + 1 / 2000.0;

        vec3 shadowNormal = texelFetch2D(shadowcolor1, ivec2(GIprojectedShadowPosition2), 0).rgb * 2.0 - 1.0;

        vec3 S1 = vec3((TC * 2.0 - 1.0) * 0.5 + 0.5, sampleDepth);
        vec4 shadowPos = shadowProjectionInverse * ((vec4(S1, 1.0) * 2.0 - 1.0) * vec4(1.0, 1.0, 7.0, 1.0));
        vec3 V = shadowPos.xyz - wpos2;

        float VdotV = dot(V, V);
        float NdotS = clamp(dot(projNormal, V * inversesqrt(VdotV)), 0.0, 1.0) * 0.8 + 0.20;
        if (NdotS <= 0.0)
            continue;

        float inHemisphere = sqrt(clamp(-dot(shadowNormal, V * inversesqrt(VdotV)), 0.0, 1.0)) * 0.8 + 0.25;
        if (inHemisphere <= 0.1)
            break;

        float Scoef = 1.0 / (1.0 + VdotV);

        gi += inHemisphere * falloff * NdotS * Scoef * pow(weight, 1.5) * sAlbedo;
    }

    const float dS = pow(576 / 512.0, 2.0);
    return ((gi * (512.0 * 512.0 / PI / steps) * giDistanceMask) * dS);
}