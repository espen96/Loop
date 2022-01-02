varying vec4 lmtexcoord;

vec3 decode(vec2 encn)
{
    vec3 unenc = vec3(0.0);
    encn = encn * 2.0 - 1.0;
    unenc.xy = abs(encn);
    unenc.z = 1.0 - unenc.x - unenc.y;
    unenc.xy = unenc.z <= 0.0 ? (1.0 - unenc.yx) * sign(encn) : encn;
    return normalize(unenc.xyz);
}

vec2 decodeVec2(float a)
{
    int bf = int(a * 65535.);
    return vec2(bf - (256 * (bf / 256)), bf >> 8) / 255.;
}

float encodeVec2v2(vec2 a)
{
    ivec2 bf = ivec2(a * 255.);
    return float(bf.x | (bf.y << 8)) / 65535.;
}
