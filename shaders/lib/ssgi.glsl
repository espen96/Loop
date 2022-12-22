
#define viewMAD(m, v) (mat3(m) * (v) + (m)[3].xyz)

vec3 RT(vec3 dir, vec3 position, float noise, vec3 N, float transparent, vec2 lightmap, bool emissive, bool hand, float z) {

    float ssptbias = SSPTBIAS;
    float depthmask = ld(z);
    float stepSize = STEP_LENGTH * clamp(blueNoise(), 0.5, 1.5);
//	int maxSteps =   clamp(int(  (clamp((moment.x),0,1)*99)),6,99);
#if STEPS != Unlimited
    int maxSteps = STEPS;
#endif
    //	if (emissive) return vec3(1.1);
    if (hand)
        return vec3(1.1);

    bool istranparent = transparent > 0.0;
    vec3 clipPosition = toClipSpace3(position);

    float rayLength = ((position.z + dir.z * sqrt(3.0) * far) > -sqrt(3.0) * near) ? (-sqrt(3.0) * near - position.z) / dir.z : sqrt(3.0) * far;

    vec3 end = toClipSpace3(position + dir * rayLength);
    vec3 direction = end - clipPosition; // convert to clip space

    float len = max(abs(direction.x) / texelSize.x, abs(direction.y) / texelSize.y) / stepSize;

    // get at which length the ray intersects with the edge of the screen
    vec3 maxLengths = (step(0., direction) - clipPosition) / direction;
    float mult = min(min(maxLengths.x, maxLengths.y), maxLengths.z);
    vec3 stepv = direction / len;
#if STEPS == Unlimited
    int iterations = int(min(len, mult * len) - 2);
#else
    int iterations = min(int(min(len, mult * len) - 2), maxSteps);
#endif

    // Do one iteration for closest texel (good contact shadows)
    vec3 spos = clipPosition * vec3(RENDER_SCALE, 1.0) + stepv / stepSize * 6.0;

    spos.xy += TAA_Offset * texelSize * 0.5 * RENDER_SCALE;

    float sp = sqrt(texelFetch2D(colortex4, ivec2(spos.xy / texelSize / 4), 0).w / 65000.0);
    float currZ = linZ(spos.z);

    if (sp < currZ) {
        float dist = abs(sp - currZ) / currZ;

        if (dist <= 0.035)
            return vec3(spos.xy, invLinZ(sp)) / vec3(RENDER_SCALE, 1.0);
    }

    stepv *= vec3(RENDER_SCALE, 1.0);

    spos += stepv * noise;
    //	for (uint i = uint(0); i < uint(8); ++i) {
    for (int i = 0; i < iterations; i++) {
        if (clamp(clipPosition.xy, 0, 1) != clipPosition.xy)
            break;
        // decode depth buffer
        float sp = sqrt(texelFetch2D(colortex4, ivec2(spos.xy / texelSize / 4), 0).w / 65000.0);

        float currZ = linZ(spos.z);
        //	if( sp < currZ && abs(sp-ld(spos.z))/ld(spos.z) < 0.1) {
        if (sp < currZ) {
            if (spos.x < 0.0 || spos.y < 0.0 || spos.z < 0.0 || spos.x > 1.0 || spos.y > 1.0 || spos.z > 1.0)
                return vec3(1.1);
            float dist = abs(sp - currZ) / currZ;

            if (dist <= ssptbias)
                return vec3(spos.xy, invLinZ(sp)) / vec3(RENDER_SCALE, 1.0);
        }

        spos += stepv;

        gl_FragData[5].r = 1;
    }

    return vec3(1.1);
}

vec3 cosineHemisphereSample(vec2 a) {
    float phi = a.y * 2.0 * 3.14159265359;
    float cosTheta = 1.0 - a.x;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
vec3 TangentToWorld(vec3 N, vec3 H) {
    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(UpVector, N));
    vec3 B = cross(N, T);

    return vec3((T * H.x) + (B * H.y) + (N * H.z));
}
vec3 toClipSpace3Prev(vec3 viewSpacePosition) {
    return projMAD(gbufferPreviousProjection, viewSpacePosition) / -viewSpacePosition.z * 0.5 + 0.5;
}

vec3 reproject(vec3 sceneSpace, bool hand) {
    vec3 prevScreenPos = hand ? vec3(0.0) : cameraPosition - previousCameraPosition;
    prevScreenPos = sceneSpace + prevScreenPos;
    prevScreenPos = viewMAD(gbufferPreviousModelView, prevScreenPos);
    prevScreenPos = viewMAD(gbufferPreviousProjection, prevScreenPos) * (0.5 / -prevScreenPos.z) + 0.5;

    return prevScreenPos;
}

vec2 moment(ivec2 pos) {

    float weightSum = 1.0;
    vec2 moment = texelFetch(colortex15, pos, 0).rg;
    float depth = texelFetch(depthtex0, pos, 0).x;
    vec3 normal = texelFetch(colortex10, pos, 0).xyz;
    for (int i = 0; i < 9; i++) {
        ivec2 deltaPos = kernelO_3x3[i] * 2;

        // ⬇️ Sample current point data with current uv
        ivec2 p = pos + deltaPos;
        vec4 curColor = texelFetch(colortex12, ivec2(p), 0);
        float curDepth = texelFetch(depthtex0, p, 0).x;
        vec3 curNormal = texelFetch(colortex10, p, 0).xyz;

        //  Determine the average brightness of this sample
        //  Using International Telecommunications Union's ITU BT.601 encoding params
        float l = luma(curColor.rgb);

        float weightDepth = abs(curDepth - depth) / (depth * length(vec2(deltaPos)) + 1.0e-2);
        float weightNormal = pow(max(0, dot(curNormal, normal)), 32.0);
        float w = exp(-weightDepth) * weightNormal;

        weightSum += w;

        moment += vec2(l, l * l) * w;
    }

    moment /= weightSum;

    return moment;
}
void temporal(inout vec3 indirectCurrent, inout vec4 historyGData, inout vec4 indirectHistory, vec3 fragpos, vec3 normal2, float z, vec2 texcoord, bool hand, vec3 inderectNoSSGI, vec2 lightmap) {

    vec3 scenePos = viewMAD(gbufferModelViewInverse, fragpos);
    vec3 reprojection = reproject(scenePos, hand);

    bool offscreen = clamp(reprojection, 0, 1) != reprojection;
    vec3 cameraMovement = mat3(gbufferModelView) * (cameraPosition - previousCameraPosition);
    vec4 currentGData = vec4(normal2, 0);
    currentGData.rgb = currentGData.rgb;
    currentGData.a = length(fragpos);

    vec4 velocity2 = texture(colortex13, texcoord).rgba;

    velocity2.r = ((velocity2.x / ld(z) * 28) * 0.2);
    velocity2.g = ((velocity2.y / ld(z) * 28) * 0.4);

    reprojection.xy *= RENDER_SCALE;
    historyGData = texture(colortex9, reprojection.xy);
    historyGData.rgb = historyGData.rgb;
    vec2 moment = moment(ivec2(floor(reprojection.xy * vec2(viewWidth, viewHeight))));
    gl_FragData[4].rg = moment;

    float distanceDelta = distance(historyGData.a * far, currentGData.a) - abs(cameraMovement.z);

    float depthRejection = (offscreen) ? 0.0 : exp(-max(distanceDelta - 0.2, 0.0) * 3.0);

    indirectHistory = texture(colortex12, reprojection.xy);
    indirectHistory.a = clamp(indirectHistory.a, 0, 1);

    float normalWeight = sqr(clamp(dot(historyGData.rgb, currentGData.rgb), 0, 1));

    float accumulationR0 = mix(0.1, 0.02, sqrt(indirectHistory.a));

    float varianceSpatial = clamp((max(0.0, moment.y - moment.x * moment.x) * 100), 0, 1);

    varianceSpatial = accumulationR0 + varianceSpatial;
    float accumulationWeight = depthRejection;
    accumulationWeight *= normalWeight;
    accumulationWeight += clamp(((distance(indirectHistory.rgb, indirectCurrent.rgb) / luma(indirectHistory.rgb))) *
        (abs(cameraMovement.r) + abs(cameraMovement.g) + abs(cameraMovement.b)), 0, 1);
    accumulationWeight = clamp(1.0 - accumulationWeight, 0.0, 1);
    accumulationWeight = max(accumulationWeight, varianceSpatial);

    vec3 closestToCamera = vec3(texcoord, z);
    vec3 fragposition = fragpos;

    fragposition = mat3(gbufferModelViewInverse) * fragposition + gbufferModelViewInverse[3].xyz +
        (cameraPosition - previousCameraPosition);
    vec3 previousPosition = mat3(gbufferPreviousModelView) * fragposition + gbufferPreviousModelView[3].xyz;
    previousPosition = toClipSpace3Prev(previousPosition);
    vec2 velocity = previousPosition.xy - closestToCamera.xy;

    previousPosition.xy = texcoord + velocity;

    vec3 albedoCurrent0 = texture(colortex12, texcoord).rgb;
    vec3 albedoCurrent1 = texture(colortex12, texcoord + vec2(texelSize.x, texelSize.y)).rgb;
    vec3 albedoCurrent2 = texture(colortex12, texcoord + vec2(texelSize.x, -texelSize.y)).rgb;
    vec3 albedoCurrent3 = texture(colortex12, texcoord + vec2(-texelSize.x, -texelSize.y)).rgb;
    vec3 albedoCurrent4 = texture(colortex12, texcoord + vec2(-texelSize.x, texelSize.y)).rgb;
    vec3 albedoCurrent5 = texture(colortex12, texcoord + vec2(0.0, texelSize.y)).rgb;
    vec3 albedoCurrent6 = texture(colortex12, texcoord + vec2(0.0, -texelSize.y)).rgb;
    vec3 albedoCurrent7 = texture(colortex12, texcoord + vec2(-texelSize.x, 0.0)).rgb;
    vec3 albedoCurrent8 = texture(colortex12, texcoord + vec2(texelSize.x, 0.0)).rgb;

    // Assuming the history color is a blend of the 3x3 neighborhood, we clamp the history to the min and max of each
    // channel in the 3x3 neighborhood
    vec3 cMax = max(max(max(albedoCurrent0, albedoCurrent1), albedoCurrent2), max(albedoCurrent3, max(albedoCurrent4, max(albedoCurrent5, max(albedoCurrent6, max(albedoCurrent7, albedoCurrent8))))));
    vec3 cMin = min(min(min(albedoCurrent0, albedoCurrent1), albedoCurrent2), min(albedoCurrent3, min(albedoCurrent4, min(albedoCurrent5, min(albedoCurrent6, min(albedoCurrent7, albedoCurrent8))))));

    vec3 albedoPrev = texture(colortex12, reprojection.xy).rgb;
    vec3 finalcAcc = clamp(albedoPrev, cMin, cMax);

    float isclamped = (clamp(clamp(((distance(albedoPrev, finalcAcc) / luma(albedoPrev))), 0, 1), 0.0, 1)) * 0.5;
    accumulationWeight = clamp((accumulationWeight) + isclamped, 0.0, 1);

    if (!offscreen) {

        //	if (length(velocity) > 1e-6) accumulationWeight += 0.1;
        indirectHistory.rgb = mix(indirectHistory.rgb, indirectCurrent, accumulationWeight);

        indirectHistory.a = mix(0.0, clamp(indirectHistory.a + (1 / (30)), 0, 1), float(distanceDelta < 0.2));
        if (hand)
            indirectHistory.a = 1;
    } else {
        indirectHistory = vec4(mix(indirectHistory.rgb, indirectCurrent, 0.9), 0.0);
    }

    indirectCurrent = indirectHistory.rgb;
    historyGData.rgb = currentGData.rgb;
    historyGData.a = currentGData.a / far;

    indirectCurrent = indirectCurrent;
    historyGData = historyGData;
    indirectHistory = indirectHistory;
}

//////////////////////////////SSGI//////////////////////////////
const float PI = acos(-1.0);
const float TAU = PI * 2.0;
const float hPI = PI * 0.5;
const float rPI = 1.0 / PI;
const float rTAU = 1.0 / TAU;

const float PHI = sqrt(5.0) * 0.5 + 0.5;
const float rLOG2 = 1.0 / log(2.0);

const float goldenAngle = TAU / PHI / PHI;
vec2 fermatsSpiralGoldenN(float index, float total) {
    float theta = index * goldenAngle;
    return vec2(sin(theta), cos(theta)) * sqrt(index / total);
}

vec2 sincos(float x) {
    return vec2(sin(x), cos(x));
}
vec3 genUnitVector(vec2 p) {
    p.x *= 6.283185307179586;
    p.y = p.y * 2.0 - 1.0;
    return vec3(sincos(p.x) * sqrt(1.0 - p.y * p.y), p.y);
}
vec3 GenerateCosineVectorSafe(vec3 vector, vec2 xy) {
    // Apparently this is actually this simple.
    // http://www.amietia.com/lambertnotangent.html
    // (cosine lobe around vector = lambertian BRDF)
    // This one deals with ther rare case where cosineVector == (0,0,0)
    // Can just normalize it instead if you are sure that won't be a problem
    vec3 cosineVector = vector + genUnitVector(xy);
    float lenSq = dot(cosineVector, cosineVector);
    return lenSq > 0.0 ? cosineVector * inversesqrt(lenSq) : vector;
}
#define HASHSCALE3 vec3(.1031, .1030, .0973)

vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * HASHSCALE3);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.xx + p3.yz) * p3.zy);
}
vec3 sphereMap(vec2 a) {
    float phi = a.y * 2.0 * PI;
    float cosTheta = 1.0 - a.x;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float ditherBluenoise() {
    ivec2 coord = ivec2(fract(gl_FragCoord.xy / 256.0) * 256.0);
    float noise = texelFetch(noisetex, coord, 0).a;
    noise = fract(noise + float(frameCounter) / PI);

    return noise;
}
const float rho = 1.32471795724474602596090885447809; // plastic constant
vec3 rtGI(vec3 normal, vec3 normal2, vec4 noise, vec3 fragpos, float translucent, vec3 ambient, vec3 torch, vec3 albedo, float z, vec2 lightmap, bool emissive, bool hand, vec2 texcoord) {

    vec3 sceneNormal = normal2 * 2.0 - 1.0;
    vec3 viewNormal = mat3(gbufferModelView) * sceneNormal;

    vec4 historyGData = vec4(1.0);
    vec4 indirectHistory = vec4(0.0);
    float sceneDepth = texture(depthtex0, texcoord.xy).x;

    vec3 viewPos = screenToViewSpace(vec3(coord, sceneDepth), gbufferProjectionInverse);
    vec3 viewDir = normalize(viewPos);

    vec3 scenePos = viewMAD(gbufferModelViewInverse, fragpos);

    vec3 reprojection = reproject(scenePos, hand);

    bool offscreen = clamp(reprojection, 0, 1) != reprojection;

    indirectHistory = texture(colortex12, reprojection.xy * RENDER_SCALE);
    indirectHistory.a = clamp(indirectHistory.a, 0, 1);

    int nrays = RAY_COUNT;
    //	int nrays = RAY_COUNT + int(4*(1-indirectHistory.a));
    //		nrays = clamp(1 * clamp(int(moment.y*1000),0,4),2,4);
    if (emissive)
        nrays = 2;
    //	int nrays = clamp(int(clamp((lightmap.x),0,1)*6),2,6);
    //	if (z > 0.50) nrays = 2;
    //	if (z > 0.75) nrays = 1;
    float mixer = SSPTMIX1;
    //	float mixer = SSPTMIX1*clamp(1-(z*10),0,1);
    float rej = 1;
    vec3 intRadiance = vec3(0.0);

    float occlusion = 0.0;
    float depthmask = ((z * z * z) * 2);
    if (depthmask > 1)
        nrays = 1;

    for (int i = 0; i < nrays; i++) {

        int seed = (frameCounter % 40000) * nrays + i;
        vec2 ij = fract(R2_samples(seed) + noise.rg);
        vec3 rayDir = normalize(cosineHemisphereSample(ij));

        rayDir = TangentToWorld(normal, rayDir);
        rayDir = mat3(gbufferModelView) * rayDir;
        if (dot(rayDir, normal2) < 0.0)
            rayDir = -rayDir;
        // rayDir = reflect(normalize(fragpos), normal);

        //	vec3 rayHit = RT2(rayDir, fragpos, fract(seed/1.6180339887 + noise.b));
        //	vec3 rayHit = RT3(rayDir, fragpos, fract(seed/1.6180339887 + noise.b));
        vec3 rayHit = RT(rayDir, fragpos, fract(seed / 1.6180339887 + noise.b), mat3(gbufferModelView) * normal, 0, lightmap, emissive, hand, z);

        vec3 previousPosition = reproject(mat3(gbufferModelViewInverse) * toScreenSpace(rayHit) + gbufferModelViewInverse[3].xyz, false);

        if (rayHit.z < 1.0) {

            if (previousPosition.x > 0.0 && previousPosition.y > 0.0 && previousPosition.x < 1.0 &&
                previousPosition.x < 1.0) {

                //	if(!hand)	intRadiance += ((texture(colortex5,previousPosition.xy).rgb*(1+(lightmap.x*2)))  +
                //ambient*albedo*translucent) ;
                if (!hand)
                    intRadiance += ((texture(colortex5, previousPosition.xy).rgb) + ambient * albedo * translucent);

#ifdef ssgi_staturation
                float lum = luma(intRadiance);
                vec3 diff = intRadiance - lum;
                intRadiance = (intRadiance + diff * (0.1));
#endif
            } else {

                intRadiance += ambient + ambient * translucent * albedo;
            }

            occlusion += 1.0;
        } else {
            float bounceAmount = float(rayDir.y > 0.0);
            vec3 sky_c = skyCloudsFromTex(rayDir, colortex4).rgb * 8 / 3. / 150. * bounceAmount;
            //	 sky_c *= clamp(eyeBrightnessSmooth.y/255.0 + lightmap.y,0.0,1.0);
            sky_c *= lightmap.y;

            intRadiance += sky_c;
        }
    }

    if (hand)
        occlusion = 0.0;

#ifdef NETHER
    intRadiance.rgb = (intRadiance / nrays + (1.0 - (occlusion) / nrays) * (torch * 0.5));
#else
    intRadiance.rgb = (intRadiance / nrays + (1.0 - (occlusion) / nrays) * (torch * SSPTambient));
#endif

    gl_FragData[5].g = texture(colortex11, gl_FragCoord.xy * texelSize).g;

    return vec3(intRadiance).rgb * (1.0 - (occlusion * 0.3) / nrays);
}
