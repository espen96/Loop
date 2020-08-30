#version 120

uniform float viewWidth;
uniform float viewHeight;

uniform int frameCounter;
varying vec2 texcoord;
uniform sampler2D colortex0;//clouds
uniform sampler2D colortex1;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex2;//albedo(rgb),material(alpha) RGBA16
uniform sampler2D colortex4;//Skybox
uniform sampler2D colortex3;
uniform sampler2D colortex5;
uniform sampler2D colortex6;//Skybox
uniform sampler2D depthtex2;//depth
uniform sampler2D depthtex1;//depth
uniform sampler2D depthtex0;//depth
uniform sampler2D noisetex;//depth
uniform sampler2D texture;
uniform sampler2D normals;
uniform vec2 texelSize;
uniform float aspectRatio;

#include "/lib/projections.glsl"

#define M_PI 3.14159
#define drawSphere true

const float radius = 0.3;

float angVelocity = -0.3,
	  phi = 0.0,
      psi = 0.0,
      theta = 0.0;


vec3 sphericalToWorld(vec2 sphCoord, float r)
{
    return vec3(
    	r * sin(sphCoord.y) * cos(sphCoord.x),
        r * sin(sphCoord.y) * sin(sphCoord.x),
        r * cos(sphCoord.y)
    );
}

vec2 worldToSpherical(vec3 flatCoord, float r)
{
    return vec2(
        atan(flatCoord.x, flatCoord.y),
        acos(flatCoord.z / r)
    );   
}
 
mat3 makeRotationMatrix(vec3 a)
{
    return mat3(
    	cos(a.x) * cos(a.z) - sin(a.x) * cos(a.y) * sin(a.z),
        -cos(a.x) * sin(a.z) - sin(a.x) * cos(a.y) * cos(a.z),
        sin(a.x) * sin(a.y),
        sin(a.x) * cos(a.z) + cos(a.x) * cos(a.y) * sin(a.z),
        -sin(a.x) * sin(a.z) + cos(a.x) * cos(a.y) * cos(a.z),
        -cos(a.x) * sin(a.y),
        sin(a.y) * sin(a.z),
        sin(a.y) * cos(a.z),
        cos(a.y)
    );
}

vec3 screenToWorld(vec2 myPos, vec2 sphereCenter, float r)
{
    vec3 myVec;
    myVec.y = myPos.x - sphereCenter.x;
    myVec.z = -(myPos.y - sphereCenter.y);
    myVec.x = sqrt(r * r - myVec.z * myVec.z - myVec.y * myVec.y);
    return myVec;
}



void main() {
float z = texture2D(depthtex1,texcoord).x;
    vec3 screenSpace = vec3(texcoord, z);
    vec4 t = gbufferProjectionInverse * vec4(screenSpace * 2.0 - 1.0, 1.0);
    vec3 viewSpace = t.xyz / t.w;
    vec3 playerSpace = mat3(gbufferModelViewInverse) * viewSpace;
    vec3 feetPlayerSpace = playerSpace + gbufferModelViewInverse[3].xyz;
    vec3 worldSpace = feetPlayerSpace + cameraPosition;
    vec2 mouse = vec2(0.0);
        
  	phi = 1 * 2.0 * M_PI / viewWidth;
    psi = 1000 * M_PI / viewHeight;
 // theta = frameCounter/100 * angVelocity;
    theta = angVelocity;
    
    vec2 sphCenter = 0.5*vec2(viewWidth/viewHeight, 1.0);
    vec2 p = gl_FragCoord.xy / vec2(viewWidth,viewHeight);
 
    vec3 worldSphCoord = sphericalToWorld(p * vec2(2.0 * M_PI, M_PI), 1.0);
    
    p.x *= viewWidth / viewHeight;
    
    if (drawSphere && length(p - sphCenter) < radius)
        worldSphCoord = screenToWorld(p, sphCenter, radius);
    
    mat3 rotationMatrix = makeRotationMatrix(vec3(phi, psi, theta));
    vec3 rotatedWorldSphCoord = normalize(rotationMatrix * worldSphCoord);

    vec2 rotatedSphericalCoord = worldToSpherical(rotatedWorldSphCoord, 1.0);

    gl_FragData[0] = texture2D(colortex3, rotatedSphericalCoord / vec2(2.0*M_PI, M_PI), 0.0);--
	
/* DRAWBUFFERS:3 */	
}