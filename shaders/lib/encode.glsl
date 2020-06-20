varying vec4 lmtexcoord;



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


//encode normal in two channels (xy),torch(z) and sky lightmap (w)
vec4 encode (vec3 n)
{

    return vec4(n.xy * inversesqrt(n.z * 8.0 + 8.0 + 0.00001) + 0.5,vec2(lmtexcoord.z,lmtexcoord.w));
}



//encoding by jodie
float encodeVec2(vec2 a){
    const vec2 constant1 = vec2( 1., 256.) / 65535.;
    vec2 temp = floor( a * 252. );
	return temp.x*constant1.x+temp.y*constant1.y;
}
float encodeVec2(float x,float y){
    return encodeVec2(vec2(x,y));
}



float encode2Vec2(vec2 a){
    const vec2 constant1 = vec2( 1., 256.) / 65535.;
    vec2 temp = ( a * 252. );
	return temp.x*constant1.x+temp.y*constant1.y;
}
float encode2Vec2(float x,float y){
#ifdef LIGHTMAP_FILTER
    return encode2Vec2(vec2(x,y));
#else	
    return encodeVec2(vec2(x,y));
#endif	
}