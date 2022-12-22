
//#define DLM
//#define POM
//#define USE_LUMINANCE_AS_HEIGHTMAP	//Can generate POM on any texturepack (may look weird in some cases)


#extension GL_EXT_gpu_shader4 : enable



#ifndef USE_LUMINANCE_AS_HEIGHTMAP
#ifndef MC_NORMAL_MAP
#undef POM
#endif
#endif
uniform int renderStage;
out float mcentity;
// 0 Undefined
// 1  Sky
// 2  Sunset and sunrise overlay
// 3  Custom sky
// 4  Sun
// 5  Moon
// 6  Stars
// 7  Void
// 8  Terrain solid
// 9  Terrain cutout mipped
// 10 Terrain cutout
// 11 Entities
// 12 Block entities
// 13 Destroy overlay
// 14 Selection outline
// 15 Debug renderers
// 16 Solid handheld objects
// 17 Terrain translucent
// 18 Tripwire string
// 19 Particles
// 20 Clouds
// 21 Rain and snow
// 22 World border
// 23 Translucent handheld objects
#ifdef POM
#define MC_NORMAL_MAP
#endif
// in vec3 at_velocity;
out vec3 velocity;
/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/

out vec4 lmtexcoord;
out vec4 color;
out vec4 normalMat;
out vec4 hspec;
out float nonlabemissive;
#if defined(POM) || defined(DLM)
out vec4 vtexcoordam; // .st for add, .pq for mul
out vec4 vtexcoord;
#endif

#ifdef MC_NORMAL_MAP
out vec4 tangent;
in vec4 at_tangent;
#endif

uniform float frameTimeCounter;
const float PI48 = 150.796447372 * WAVY_SPEED;
float pi2wt = PI48 * frameTimeCounter;
in vec4 mc_Entity;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
in vec4 mc_midTexCoord;
uniform vec3 cameraPosition;
out vec2 taajitter;
uniform vec2 texelSize;
uniform int framemod8;
const vec2[8] offsets = vec2[8](vec2(1. / 8., -3. / 8.), vec2(-1., 3.) / 8., vec2(5.0, 1.) / 8., vec2(-3, -5.) / 8.,
                                vec2(-5., 5.) / 8., vec2(-7., -1.) / 8., vec2(3, 7.) / 8., vec2(7., -7.) / 8.);
#define diagonal3(m) vec3((m)[0].x, (m)[1].y, m[2].z)
#define projMAD(m, v) (diagonal3(m) * (v) + (m)[3].xyz)
vec4 toClipSpace3(vec3 viewSpacePosition)
{
    return vec4(projMAD(gl_ProjectionMatrix, viewSpacePosition), -viewSpacePosition.z);
}
#ifdef WAVY_PLANTS
vec2 calcWave(in vec3 pos)
{

    float magnitude = abs(sin(dot(vec4(frameTimeCounter, pos), vec4(1.0, 0.005, 0.005, 0.005))) * 0.5 + 0.72) * 0.013;
    vec2 ret = (sin(pi2wt * vec2(0.0063, 0.0015) * 4. - pos.xz + pos.y * 0.05) + 0.1) * magnitude;

    return ret;
}

vec3 calcMovePlants(in vec3 pos)
{
    vec2 move1 = calcWave(pos);
    float move1y = -length(move1);
    return vec3(move1.x, move1y, move1.y) * 5. * WAVY_STRENGTH;
}

vec3 calcWaveLeaves(in vec3 pos, in float fm, in float mm, in float ma, in float f0, in float f1, in float f2,
                    in float f3, in float f4, in float f5)
{

    float magnitude = abs(sin(dot(vec4(frameTimeCounter, pos), vec4(1.0, 0.005, 0.005, 0.005))) * 0.5 + 0.72) * 0.013;
    vec3 ret = (sin(pi2wt * vec3(0.0063, 0.0224, 0.0015) * 1.5 - pos)) * magnitude;

    return ret;
}

vec3 calcMoveLeaves(in vec3 pos, in float f0, in float f1, in float f2, in float f3, in float f4, in float f5,
                    in vec3 amp1, in vec3 amp2)
{
    vec3 move1 = calcWaveLeaves(pos, 0.0054, 0.0400, 0.0400, 0.0127, 0.0089, 0.0114, 0.0063, 0.0224, 0.0015) * amp1;
    return move1 * 5. * WAVY_STRENGTH;
}
#endif
float luma(vec3 color)
{
    return dot(color, vec3(0.21, 0.72, 0.07));
}
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////
//////////////////////////////VOID MAIN//////////////////////////////

void main()
{

    hspec = hspec = vec4(50, 250, 000, 000);
    nonlabemissive = 0.0;

    if (mc_Entity.x == 10005)
        nonlabemissive = 0.1;
    if (mc_Entity.x == 2000)
        nonlabemissive = 0.2;
    if (mc_Entity.x == 2001)
        nonlabemissive = 0.3;
    if (mc_Entity.x == 2002)
        nonlabemissive = 0.4;
    if (mc_Entity.x == 2003)
        nonlabemissive = 0.5;
    if (mc_Entity.x == 2004)
        nonlabemissive = 0.6;
    if (mc_Entity.x == 2005)
        nonlabemissive = 0.7;

    if (mc_Entity.x == 1101)
        hspec = vec4(250, 250, 000, 000);
    if (mc_Entity.x == 1102)
        hspec = vec4(250, 10, 000, 000);
    if (mc_Entity.x == 1103)
        hspec = vec4(200, 10, 000, 000);
    if (mc_Entity.x == 1104)
        hspec = vec4(100, 20, 000, 000);

    lmtexcoord.xy = gl_MultiTexCoord0.xy;
#if defined(POM) || defined(DLM)
    vec2 midcoord = mc_midTexCoord.st;
    vec2 texcoordminusmid = lmtexcoord.xy - midcoord;
    vtexcoordam.pq = abs(texcoordminusmid) * 2;
    vtexcoordam.st = min(lmtexcoord.xy, midcoord - texcoordminusmid);
    vtexcoord.xy = sign(texcoordminusmid) * 0.5 + 0.5;
#endif
    vec2 lmcoord = gl_MultiTexCoord2.xy / 255.0;
    lmtexcoord.zw = lmcoord;

    vec3 position = mat3(gl_ModelViewMatrix) * gl_Vertex.xyz + gl_ModelViewMatrix[3].xyz;

    color = gl_Color;
    taajitter = offsets[framemod8];
    hspec *= clamp(1 - luma(color.rgb * 10 - 5), 0.95, 1);
    bool istopv = gl_MultiTexCoord0.t < mc_midTexCoord.t;
#ifdef MC_NORMAL_MAP
    tangent = vec4(normalize(gl_NormalMatrix * at_tangent.rgb), at_tangent.w);
#endif

    normalMat =
        vec4(normalize(gl_NormalMatrix * gl_Normal),
             mc_Entity.x == 10004 || mc_Entity.x == 10003 || mc_Entity.x == 80 || mc_Entity.x == 10001 ? 0.5 : 1.0);
    normalMat.a = mc_Entity.x == 10006 ? 0.6 : normalMat.a;

#ifdef WAVY_PLANTS
    if ((mc_Entity.x == 10001 && istopv) && abs(position.z) < 64.0)
    {
        vec3 worldpos = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz + cameraPosition;
        worldpos.xyz += calcMovePlants(worldpos.xyz) * lmtexcoord.w - cameraPosition;
        position = mat3(gbufferModelView) * worldpos + gbufferModelView[3].xyz;
    }

    if (mc_Entity.x == 10003 && abs(position.z) < 64.0)
    {
        vec3 worldpos = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz + cameraPosition;
        worldpos.xyz += calcMoveLeaves(worldpos.xyz, 0.0040, 0.0064, 0.0043, 0.0035, 0.0037, 0.0041,
                                       vec3(1.0, 0.2, 1.0), vec3(0.5, 0.1, 0.5)) *
                            lmtexcoord.w -
                        cameraPosition;
        position = mat3(gbufferModelView) * worldpos + gbufferModelView[3].xyz;
    }
#endif
    if (mc_Entity.x == 1 || mc_Entity.x == 2 || mc_Entity.x == 3 || mc_Entity.x == 4 || mc_Entity.x == 5 ||
        mc_Entity.x == 6 || mc_Entity.x == 7 || mc_Entity.x == 8 || mc_Entity.x == 9 || mc_Entity.x == 10 ||
        mc_Entity.x == 11 || mc_Entity.x == 12 || mc_Entity.x == 13 || mc_Entity.x == 14 || mc_Entity.x == 15)
    {
        color.rgb = normalize(color.rgb) * sqrt(3.0);
        normalMat.a = 0.9;
    }
    gl_Position = toClipSpace3(position);
#ifdef SEPARATE_AO
    lmtexcoord.zw *= vec2(sqrt(color.a),color.a);
#else
    color.rgb *= color.a;
#endif

#ifdef TAA_UPSCALING
    gl_Position.xy = gl_Position.xy * RENDER_SCALE + RENDER_SCALE * gl_Position.w - gl_Position.w;
#endif
#ifdef TAA
    gl_Position.xy += offsets[framemod8] * gl_Position.w * texelSize;
#endif

    //	velocity = at_velocity / 1000000.0;
    velocity = vec3(0.0);

    mcentity = 0;
    if (mc_Entity.x < 16)
        mcentity = mc_Entity.x;
}
