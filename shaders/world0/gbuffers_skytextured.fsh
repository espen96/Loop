#version 120
#extension GL_EXT_gpu_shader4 : enable
/*
!! DO NOT REMOVE !!
This code is from Chocapic13' shaders
Read the terms of modification and sharing before changing something below please !
!! DO NOT REMOVE !!
*/

/* DRAWBUFFERS:1 */
varying vec3 viewPosition;
uniform vec3 moonDirection;
varying vec4 color;
varying vec2 texcoord;
uniform mat4 gbufferModelView;
uniform int moonPhase;
#include "/lib/util.glsl"
uniform mat4 shadowModelViewInverse;
uniform sampler2D noisetex;//depth
uniform sampler2D colortex4;//depth
uniform sampler2D texture;
uniform int worldTime;



	float SmoothMoon(){
		return  (float(moonPhase) + float(worldTime) * 0.000041666666666666) * 0.125;
	}


	vec2 SphereIntersection(vec3 position, vec3 direction, float radius) {
		float PoD = dot(position, direction);
		float radiusSquared = radius * radius;

		float delta = PoD * PoD + radiusSquared - dot(position, position);
		if (delta < 0.0) return vec2(-1.0);
		      delta = sqrt(delta);

		return -PoD + vec2(-delta, delta);
	}

	vec3 Rotate(vec3 vector, vec3 axis, float angle) {

		float cosine = cos(angle);
		float sine = sin(angle);

		float tmp = dot(axis, vector);
		return cosine * vector + sine * cross(axis, vector) + (tmp - tmp * cosine) * axis;
	}									
	float fbm(vec2 p) {

		return texture2D(noisetex, p).x*0.5;
	}		

void main() {



		vec3 viewDirection = normalize(viewPosition);

		vec3 sc = -texelFetch2D(colortex4,ivec2(6,37),0).rgb;
	


			const float angle = TAU * (1.0 - cos(atan(0.05)));

			const vec2    scale  = vec2(1.0 / 4.0, 1.0 / 2.0);
			const vec2[8] offset = vec2[8](
				vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(2.0, 0.0), vec2(3.0, 0.0),
				vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(2.0, 1.0), vec2(3.0, 1.0)
			);

			vec2 uv = (texcoord / scale) - offset[moonPhase];
			     uv = uv * 2.0 - 1.0;			
				 if (abs(uv.x) > 0.25 || abs(uv.y) > 0.25) { discard; }
			float noise = fbm(uv);
			vec4 col = vec4(dot(uv, uv) <= 0.10*0.10);

			vec2 t = SphereIntersection(-moonDirection, viewDirection, asin(0.02));
			vec3 normal = normalize(t.x * viewDirection - moonDirection)*(1-noise);
			vec3 light = Rotate(-moonDirection, mat3(gbufferModelView) * shadowModelViewInverse[0].xyz, -TAU *  SmoothMoon());

			col.rgb *= 0.04 * max(dot(normal, light), 0.0) + sc*0.2 * max(dot(normal, -moonDirection), 0.0);


			col.rgb *=   0.2 / angle;
		
		
	gl_FragData[0] = col;

}
