shader_type canvas_item;

uniform float velocity_max = 10.0;
uniform sampler2D color_ramp;

varying vec3 particle_color;

void vertex() {
    // The color is set per instance via MultiMesh
    particle_color = COLOR.rgb;
}

void fragment() {
    // Calculate distance from center
    vec2 center_offset = (UV - 0.5) * 2.0;
    float sqr_dst = dot(center_offset, center_offset);

    // Smooth circle edge
    float delta = fwidth(sqrt(sqr_dst));
    float alpha = 1.0 - smoothstep(1.0 - delta, 1.0 + delta, sqr_dst);

    COLOR = vec4(particle_color, alpha);
}