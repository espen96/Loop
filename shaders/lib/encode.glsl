vec3 decode (vec2 enc)
{
    vec2 fenc = enc* 4.0 - 2.0;
    float f = dot(fenc,fenc);
    float g = sqrt(1.0 - f * 0.25);
return vec3(fenc * g, 1.0 - f * 0.5);
}


vec2 decodeVec2(float a){
    const vec2 constant1 = 65536. / vec2( 256.0, 65536.);
    const float constant2 = 256. / 255.;
    return fract( a * constant1 ) * constant2 ;
}

