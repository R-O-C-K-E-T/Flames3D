#version 430
out vec2 texCoord;

const vec2 verts[4] = {vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1)};
void main() {
    gl_Position = vec4(verts[gl_VertexID], 0, 1);
	texCoord = (verts[gl_VertexID] + 1) / 2;
}