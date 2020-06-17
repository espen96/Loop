#version 300 es
precision highp float;

in vec2 UV;
out vec4 out_color;
uniform float ratio, time;
uniform sampler2D texture0;

const float PI_3 = 1.0471975512;

void main(void) {
    
    float v = 0.004f;
    vec2 d = vec2(v / ratio, v);

    #define hexa(k) vec2(cos(PI_3 * k), sin(PI_3 * k))
    vec2 deltas[6] = vec2[6](
	hexa(0.), hexa(1.), hexa(2.), 
        hexa(3.), hexa(4.), hexa(5.) 
    );
    #undef hexa

    vec2 xy = vec2(UV.x, 1.0f - UV.y);
    vec4 col = texture(texture0,  xy);

    for (int i = 0; i < 6; ++i) {
        vec4 col2 = texture(texture0,  xy + deltas[i] * d);
        vec4 t = max(sign(col2 - col), 0.);
        col += (col2 - col) * t;
    }

    col.a = 1.0;
    out_color = col;
}
