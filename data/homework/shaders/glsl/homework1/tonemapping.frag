#version 450

// layout (set = 0, binding = 0) uniform sampler2D samplerColor;
layout (input_attachment_index = 0, binding = 0) uniform subpassInput samplerColor;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main()
{
    vec3 c = subpassLoad(samplerColor).rgb;
	vec3 a = c * (c + 0.0245786) - 0.000090537;
	vec3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
    outFragColor = vec4(a / b, 1.0);
}