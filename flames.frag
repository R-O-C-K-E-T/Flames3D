#version 430
in vec2 texCoord;

uniform usampler3D histogram;
uniform sampler3D colour;
uniform float factor;
uniform float crossSection[DEPTH];

const float PI = 3.14159265358979;

void main() {
	vec3 finalColour = vec3(0,0,0);
	//for (int layer = 0; layer < DEPTH; layer++) {
	for (int layer = DEPTH-1; layer >= 0; layer--) {
		//uint hits = textureOffset(histogram, vec3(texCoord, 0), ivec3(0,0,layer)).r;
		uint hits = texture(histogram, vec3(texCoord, float(layer) / DEPTH)).r;

		if (hits == 0) continue;

		//vec3 colour = textureOffset(colour, vec3(texCoord, 0), ivec3(0,0,layer)).rgb;
		vec3 colour = texture(colour, vec3(texCoord, float(layer) / DEPTH)).rgb;

		float alpha = exp(factor * crossSection[layer] * float(hits));
		finalColour = finalColour * alpha + (1-alpha)*colour;
		//finalColour = colour;
		//break;
	}
	
	gl_FragColor = vec4(finalColour, 1);
}
