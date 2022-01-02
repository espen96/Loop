
float getWaterHeightmap(vec2 posxz, float iswater)
{
    vec2 pos = posxz;
    float moving = clamp(iswater * 2. - 1.0, 0.0, 1.0);
    vec2 movement = vec2(-0.005 * frameTimeCounter * moving);
    float caustic = 0.0;
    float weightSum = 0.0;
    float radiance = 2.39996;
    mat2 rotationMatrix = mat2(vec2(cos(radiance), -sin(radiance)), vec2(sin(radiance), cos(radiance)));
    pos = pos * vec2(3., 1.0) / 128. + movement;
    for (int i = 0; i < 3; i++)
    {
        float wave = texture(noisetex, pos * exp(i)).b;
        caustic += wave * exp(-i);
        weightSum += exp(-i);
        pos = rotationMatrix * pos;
    }
    return caustic / weightSum;
}
