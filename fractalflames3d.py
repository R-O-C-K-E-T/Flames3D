import argparse, pygame, time, math, random, re, json
from PIL import Image
from os import path

import numpy as np

from ctypes import *

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GL import shaders

HELPERS = '''// RANDOM = uniform random float in [0,1)
// RAND_VEC = vec3(RANDOM, RANDOM, RANDOM)
const float radius2 = dot(coord, coord);
const float radius = sqrt(radius2);
const float theta = atan(coord.x, coord.y);
const float phi = (PI/2) - theta;'''

def convert_from_linear(colour):
    return [u*(323/25) if u <= 0.0031308 else (211*u**(5/12) - 11)/200 for u in colour]

def convert_to_linear(colour):
    return [u*(25/323) if u <= 0.04045 else ((200*u+11)/211)**(12/5) for u in colour]

def projection_mat(aspect, near, far, fovy):
    f = 1 / math.tan(math.radians(fovy) / 2)
    a = (far + near) / (near - far)
    b = 2 * far * near / (near - far)

    return np.array([
        [f/aspect, 0,  0, 0],
        [0,        f,  0, 0],
        [0,        0,  a, b],
        [0,        0, -1, 0],
    ], np.float32)

def translation_mat(offset):
    x, y, z = offset
    return np.array([[1,0,0,x],
                     [0,1,0,y],
                     [0,0,1,z],
                     [0,0,0,1],],np.float32)

def rotationX_mat(angle):
    a, b, c = math.cos(angle), math.sin(angle), -math.sin(angle)
    return np.array([[1,0,0,0],
                     [0,a,b,0],
                     [0,c,a,0],
                     [0,0,0,1],],np.float32)

def rotationY_mat(angle):
    a, b, c = math.cos(angle), math.sin(angle), -math.sin(angle)
    return np.array([[a,0,c,0],
                     [0,1,0,0],
                     [b,0,a,0],
                     [0,0,0,1],],np.float32)

def rotationZ_mat(angle):
    a, b, c = math.cos(angle), math.sin(angle), -math.sin(angle)
    return np.array([[a,b,0,0],
                     [c,a,0,0],
                     [0,0,1,0],
                     [0,0,0,1],],np.float32)

def rotation_mat(rot):
    x, y, z = rot
    return rotationY_mat(y) @  rotationX_mat(x) @ rotationZ_mat(z)

def compile_shader(text, shader_type, defines={}):
    def add_globals(string):
        header = ''
        for key, val in defines.items():
            if isinstance(val, bool):
                if val:
                    header += '#define {}\n'.format(key)
                else:
                    header += '#undef {}\n'.format(key)
            else:
                val = str(val).replace('\n', '\\\n')
                header += '#define {} {}\n'.format(key, val)

        index = string.index('\n')+1
        return string[:index] + header + string[index:]

    text = add_globals(text)

    #print(text)
    #print('\n\n\n\n')

    #return shaders.compileShader(text, type)
    try:
        return shaders.compileShader(text, shader_type)
    except RuntimeError as e:
        lines = text.split('\n')
        for cause in e.args[0].split('\\n'):
            print(cause)
            match = re.search('0\\(([0-9]+)\\)', cause)
            if match is None:
                continue
            line = int(match[1]) - 1
            print(*lines[line-1:line+2], sep='\n')
    raise RuntimeError("Compilation Failed")

def genTexture3D():
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_3D, texture)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    return texture

def genTexture2D():
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    return texture

def load_image(filename, texture=None):
    if texture is None:
        texture = genTexture2D()

    img = Image.open(filename)
    img = img.transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
    data = img.tobytes()
    
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, *img.size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return texture

class Particle(Structure):
    _fields_ = [
        ('pos', c_float * 3),
        ('pad1', c_float * 1),
        ('col', c_float * 4),
        ('randState', c_uint * 4),
        #('pad2', c_float * 4),
    ]

def get_texture_size(texture):
    glBindTexture(GL_TEXTURE_3D, texture)
    width = glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_HEIGHT)
    depth = glGetTexLevelParameteriv(GL_TEXTURE_3D, 0, GL_TEXTURE_DEPTH)

    return width, height, depth

def fix_image_paths(flame, base_directory):
    for func in flame['functions']:
        for variation in func['variations']:
            for name, image in variation.get('images', {}).items():
                variation['images'][name] = path.join(base_directory, image)

'''def deep_equal(a, b):
    if a is b:
        return True

    if type(a) == list and type(b) == list:
        if len(a) != len(b):
            print(a, b)
            return False

        val = all(deep_equal(item_a, item_b) for item_a, item_b in zip(a, b))
        if not val:
            print(a, b)
        return val

    if type(a) == dict and type(b) == dict:
        if len(a) != len(b):
            print(a, b)
            return False

        val = all(deep_equal(key_a, key_b) and deep_equal(val_a, val_b) for (key_a, val_a), (key_b, val_b) in zip(a.items(), b.items()))
        if not val:
            print(a, b)
        return val

    val = a == b
    if not val:
        print(a, b)
    return val'''

def check_flame_compatibility(flame_a, flame_b):
    if flame_a is flame_b:
        return True

    if flame_a is None or flame_b is None:
        return False

    functions_a = flame_a['functions']
    functions_b = flame_b['functions']

    if len(functions_a) != len(functions_b):
        return False

    for func_a, func_b in zip(functions_a, functions_b):
        variations_a = func_a['variations']
        variations_b = func_b['variations']
        if len(variations_a) != len(variations_b):
            return False

        for var_a, var_b in zip(variations_a, variations_b):
            if var_a['base'] != var_b['base']:
                return False

            if set(var_a['params']) != set(var_b['params']):
                return False

            if set(var_a.get('images', {})) != set(var_b.get('images', {})):
                return False

    return True

def check_flame_identical(flame_a, flame_b):
    if flame_a is flame_b:
        return True

    if flame_a is None or flame_b is None:
        return flame_a == flame_b

    flame_a = flame_a.copy()
    flame_b = flame_b.copy()

    for flame in flame_a, flame_b:
        del flame['exposure']

    return flame_a == flame_b

class Iterator:
    def __init__(self, histogram, colour, /, local_size=(16, 16), particles=(256, 256)):
        self.histogram = histogram
        self.colour = colour

        self.local_size = local_size
        self.particles = particles

        histogram_size = get_texture_size(histogram)
        colour_size = get_texture_size(colour)

        assert histogram_size == colour_size
        self.size = colour_size

        self.tick = 0
        self.convergence_timer = None

        self.flame = None

        self.prog = None
        self.matrix_uniform = None

        self.image_cache = {} # filename -> texture_id

        self.particle_buffers = glGenBuffers(2).tolist()
        self.reset_particles()

    def reload_images(self):
        for filename, texture_id in self.image_cache.items():
            load_image(filename, texture_id)

    @property
    def iterations_per_tick(self):
        return self.particles[0] * self.particles[1]

    def cycle(self):
        assert self.prog is not None

        if self.convergence_timer is not None:
            self.convergence_timer -= 1
            if self.convergence_timer == 0:
                self.clear()
                self.convergence_timer = None


        glUseProgram(self.prog)
        glBindImageTexture(1, self.colour, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)
        glBindImageTexture(0, self.histogram, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI)

        for i, texture in enumerate(self.image_cache.values()):
            glActiveTexture(GL_TEXTURE0 + (2 + i))
            glBindTexture(GL_TEXTURE_2D, texture)

        buffer = bool(self.tick % 2)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.particle_buffers[not buffer]) # in
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.particle_buffers[buffer]) # out

        glDispatchCompute(self.particles[0]//self.local_size[0], self.particles[1]//self.local_size[1], 1)

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        self.tick += 1

    def reset_particles(self):
        N = self.particles[0] * self.particles[1]

        positions = np.random.random((N, 3))
        random_states = np.random.randint(1<<32, size=(N, 4), dtype=np.uint32)

        particle_data = (Particle * N)()
        for particle, position, rand_state in zip(particle_data, positions, random_states):
            particle.pos[0] = position[0]
            particle.pos[1] = position[1]
            particle.pos[2] = position[2]

            particle.randState[0] = rand_state[0]
            particle.randState[1] = rand_state[1]
            particle.randState[2] = rand_state[2]
            particle.randState[3] = rand_state[3]

        for buffer in self.particle_buffers:
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferData(GL_ARRAY_BUFFER, sizeof(particle_data), particle_data, GL_STATIC_DRAW)

    def reload_program(self):
        if self.prog is not None:
            glDeleteProgram(self.prog)

        defines = {
            'FUNC_COUNT': len(self.flame['functions']),
            'LOCAL_SIZE_X': self.local_size[0],
            'LOCAL_SIZE_Y': self.local_size[1],
            'GLOBAL_SIZE_X': self.particles[0],
            'GLOBAL_SIZE_Y': self.particles[1],
            'WIDTH': self.size[0],
            'HEIGHT': self.size[1],
            'DEPTH': self.size[2],
        }

        function_defines = self._generate_function_defines()

        shader_text = open('flames.comp').read()
        shader_text = shader_text.replace('FUNCTIONS', function_defines['FUNCTIONS'])
        shader_text = shader_text.replace('APPLY_FUNCTION', function_defines['APPLY_FUNCTION'])
        shader = compile_shader(shader_text, GL_COMPUTE_SHADER, defines)


        self.prog = shaders.compileProgram(shader)
        glDeleteShader(shader)

        glUseProgram(self.prog)
        self.matrix_uniform = glGetUniformLocation(self.prog, 'viewMat')

        self.set_flame_uniforms()

    def clear(self):
        # Not neccessary to clear colour data
        #glClearTexImage(self.colour, 0, GL_RGBA, GL_FLOAT, c_float(0))
        
        glClearTexImage(self.histogram, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, c_uint(0))
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        self.tick = 0

    def set_flame_uniforms(self):
        acc = 0

        textures = {} # filename -> [uniform]

        glUseProgram(self.prog)
        for i, func in enumerate(self.flame['functions']):
            glUniformMatrix4x3fv(glGetUniformLocation(self.prog, 'preTransforms[{}]'.format(i)), 1, False, func['pre_trans'])
            glUniformMatrix4x3fv(glGetUniformLocation(self.prog, 'postTransforms[{}]'.format(i)), 1, False, func['post_trans'])

            colour = func['colour'].copy()
            if len(colour) < 4:
                colour.append(0.5)
            assert len(colour) == 4
            glUniform4fv(glGetUniformLocation(self.prog, 'colours[{}]'.format(i)), 1, colour)

            acc += func['prob']
            glUniform1f(glGetUniformLocation(self.prog, 'cutoff[{}]'.format(i)), acc)

            image_id = 0
            for var in func['variations']:
                for j, (_, value) in enumerate(sorted(var['params'].items())):
                    glUniform1f(glGetUniformLocation(self.prog, 'params{}[{}]'.format(i, j)), value)

                for _, filename in sorted(var.get('images', {}).items()):
                    uniform = glGetUniformLocation(self.prog, 'img_{}_{}'.format(i, image_id))

                    if filename not in textures:
                        textures[filename] = [uniform]
                    else:
                        textures[filename].append(uniform)

                    image_id += 1
        
        old_image_cache = self.image_cache
        self.image_cache = {}
        for i, (filename, uniforms) in enumerate(textures.items()):
            try:
                texture = old_image_cache.pop(filename)
            except KeyError:
                texture = load_image(filename)

            self.image_cache[filename] = texture

            for uniform in uniforms:
                glUniform1i(uniform, i + 2)

        glDeleteTextures(list(old_image_cache.values()))

    def set_flame(self, flame):
        prev, self.flame = self.flame, flame
        if check_flame_identical(prev, flame):
            return

        self.clear()

        if flame is None or len(flame['functions']) == 0:
            if self.prog is not None:
                glDeleteProgram(self.prog)
                self.prog = None
            return

        if self.prog is None or flame['functions'] != prev['functions']:
            if check_flame_compatibility(prev, flame):
                self.set_flame_uniforms()
            else:
                self.reload_program()
            self.convergence_timer = 20

        proj_mat = projection_mat(self.size[0] / self.size[1], flame['near'], flame['far'], flame['fov'])
        view_mat = rotation_mat(flame['rot']).T @ translation_mat(-np.array(flame['pos']))

        view_proj_mat = proj_mat @ view_mat

        glUseProgram(self.prog)
        glUniformMatrix4fv(self.matrix_uniform, 1, True, view_proj_mat)

    def dump_histogram(self):
        glBindTexture(GL_TEXTURE_3D, self.histogram)

        size = get_texture_size(self.histogram)
        return glGetTexImage(GL_TEXTURE_3D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, np.empty(size[::-1], np.uint32))
    
    def dump_colour(self):
        glBindTexture(GL_TEXTURE_3D, self.colour)

        size = get_texture_size(self.colour)
        return glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, np.empty((*size[::-1], 4), np.float32))

    def cleanup(self):
        if self.prog is not None:
            glDeleteProgram(self.prog)
            self.prog = None

        if self.particle_buffers is not None:
            glDeleteBuffers(2, self.particle_buffers)
            self.particle_buffers = None

        glDeleteTextures(list(self.image_cache.values()))
        self.image_cache = {}

    def _gen_function_text(self, i, function):
        text  = 'vec3 evaluate{}(inout uvec4 state, inout vec4 funcColour, vec3 pos) {{\n'.format(i)

        text += 'const mat4x3 preTrans = preTransforms[{}];\n'.format(i)
        text += 'const mat4x3 postTrans = postTransforms[{}];\n'.format(i)

        text += 'vec3 coord = preTrans * vec4(pos, 1);\n'
        text += HELPERS + '\n'
        text += 'vec3 result = vec3(0);\n'

        image_id = 0
        for variation in function['variations']:
            var_text = variation['base']
            for img_name in sorted(variation.get('images', {})):
                var_text = var_text.replace(img_name, 'img_{}_{}'.format(i, image_id))
                image_id += 1

            text += '{\n'
            for j, param_name in enumerate(sorted(variation['params'])):
                text += 'const float {} = params{}[{}];\n'.format(param_name, i, j)
            text += var_text + '\n'
            text += '}\n'

        text += 'return postTrans * vec4(result, 1);\n'
        text += '}\n'

        return text

    def _generate_function_defines(self):
        functions = self.flame['functions']
        if len(functions) == 0:
            raise RuntimeError('No functions')

        functions_text = 'uniform float cutoff[FUNC_COUNT];\n' + \
            'uniform vec4 colours[FUNC_COUNT];\n' + \
            'uniform mat4x3 preTransforms[FUNC_COUNT];\n' + \
            'uniform mat4x3 postTransforms[FUNC_COUNT];\n'

        for i, func in enumerate(functions):
            n_params = sum(len(var['params']) for var in func['variations'])

            if n_params != 0:
                functions_text += 'uniform float params{}[{}];\n'.format(i, n_params)

            n_textures = sum(len(var.get('images', {})) for var in func['variations'])
            for j in range(n_textures):
                functions_text += 'uniform sampler2D img_{}_{};\n'.format(i, j)

        functions_text += '\n'
        functions_text += '\n\n'.join(self._gen_function_text(i, func) for i, func in enumerate(functions))


        apply_function_text = ''
        if len(functions) == 1:
            apply_function_text  = 'funcColour = colours[0];\n'
            apply_function_text += 'pos = evaluate0(state, funcColour, partIn[particle].pos.xyz);\n'
        else:
            for i, func in enumerate(functions):
                if i != 0:
                    apply_function_text += ' else '

                if i != len(functions) - 1:
                    apply_function_text += 'if (accumulator < cutoff[{}]) '.format(i)

                apply_function_text += '{\n'

                apply_function_text += 'funcColour = colours[{}];\n'.format(i)
                apply_function_text += 'pos = evaluate{}(state, funcColour, partIn[particle].pos.xyz);\n'.format(i)

                apply_function_text += '}'

        return {'APPLY_FUNCTION': apply_function_text, 'FUNCTIONS': functions_text}

class Renderer:
    def __init__(self, histogram, colour):
        self.histogram = histogram
        self.colour = colour

        histogram_size = get_texture_size(histogram)
        colour_size = get_texture_size(colour)

        assert histogram_size == colour_size

        self.width, self.height, self.depth = colour_size

        defines = {
            'WIDTH': self.width,
            'HEIGHT': self.height,
            'DEPTH': self.depth,
        }

        vert = compile_shader(open('flames.vert').read(), GL_VERTEX_SHADER, defines)
        frag = compile_shader(open('flames.frag').read(), GL_FRAGMENT_SHADER, defines)

        self.prog = shaders.compileProgram(vert, frag)

        glDeleteShader(vert)
        glDeleteShader(frag)

        glUseProgram(self.prog)
        glUniform1i(glGetUniformLocation(self.prog, 'colour'),    0)
        glUniform1i(glGetUniformLocation(self.prog, 'histogram'), 1)

        self.factor_uniform = glGetUniformLocation(self.prog, 'factor')
        self.cross_section_uniform = glGetUniformLocation(self.prog, 'crossSection')

        self._proj_params = None

    def _compute_voxel_crosssection(self, proj_mat):
        ndc = np.zeros((self.depth, 4)) + 1
        ndc[:, 2] = (np.arange(self.depth) + 0.5) * 2 / self.depth - 1

        ndc = (np.linalg.inv(proj_mat) @ ndc.T).T

        return 4 * ndc[:, 0] * ndc[:, 1] / (self.width * self.height * ndc[:, 3] * ndc[:, 3])

    def render(self, flame, iterations):
        glUseProgram(self.prog)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_3D, self.colour)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_3D, self.histogram)

        proj_params = flame['near'], flame['far'], flame['fov']
        if self._proj_params != proj_params:
            proj_mat = projection_mat(self.width / self.height, *proj_params)
            areas = self._compute_voxel_crosssection(proj_mat)
            glUniform1fv(self.cross_section_uniform, self.depth, 1 / areas)

            self._proj_params = proj_params

        glUniform1f(self.factor_uniform, -flame['exposure'] / iterations)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    def cleanup(self):
        if self.prog is not None:
            glDeleteProgram(self.prog)
            self.prog = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a 3d fractal flame')

    parser.add_argument('flame', type=str)

    parser.add_argument("--width", type=int, default=1000, help='Image width')
    parser.add_argument("--height", type=int, default=1000, help='Image height')
    parser.add_argument("--depth", type=int, default=64, help='Depth Resolution')


    args = parser.parse_args()

    flame = json.load(open(args.flame))
    fix_image_paths(flame, path.dirname(args.flame))

    display = args.width, args.height
    pygame.display.set_mode(display, OPENGL|DOUBLEBUF)

    glViewport(0, 0, *display)
    glEnable(GL_FRAMEBUFFER_SRGB)

    glClearColor(0,0,0,1)

    clock = pygame.time.Clock()

    title = 'Fractal Flames 3D'
    pygame.display.set_caption(title)


    size = *display, args.depth

    colour = genTexture3D()
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, *size, 0, GL_RGBA, GL_FLOAT, None)
    histogram = genTexture3D()
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, *size, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, None)

    iterator = Iterator(histogram, colour)
    iterator.set_flame(flame)
    renderer = Renderer(histogram, colour)

    cycles = 70

    keys = ((K_d, K_a),(K_q,K_e),(K_s, K_w),(K_KP8,K_KP2),(K_KP4,K_KP6),(K_KP7,K_KP9),(K_LCTRL,K_LSHIFT))

    pos = np.array(flame['pos'], float)
    rot = np.array(flame['rot'], float)

    t = 0
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_r:
                        flame = json.load(open(args.flame))
                        fix_image_paths(flame, path.dirname(args.flame))

                        iterator.reload_images()

            pressed = pygame.key.get_pressed()

            action = np.array([pressed[b] - pressed[a] for a, b in keys])
            rot += action[3:6] / 30

            speed = [0.03, 0.1, 0.3][action[6]+1]
            pos -= action[0:3] @ rotationY_mat(-rot[1])[:3,:3] * speed

            did_change = np.any(action[0:6] != 0)

            pygame.display.set_caption('{}: {:.2f} ({:,}) - [{:.3f}, {:.3f}, {:.3f}]'.format(title, clock.get_fps(), round(clock.get_fps() * iterator.iterations_per_tick * cycles), *pos))
            glClear(GL_COLOR_BUFFER_BIT)

            new_flame = flame.copy()
            new_flame['pos'] = pos.tolist()
            new_flame['rot'] = rot.tolist()

            iterator.set_flame(new_flame)

            for _ in range(cycles):
                iterator.cycle()

            renderer.render(flame, iterator.tick * iterator.iterations_per_tick)

            pygame.display.flip()
            clock.tick(60)
            t += 1
    finally:
        #print(pos.tolist(), rot.tolist())

        if False:
            histogram_data = iterator.dump_histogram()
            print(np.sum(histogram_data), iterator.tick * iterator.iterations_per_tick)
            #np.save('temp', histogram_data)

        iterator.cleanup()
        renderer.cleanup()

        glDeleteTextures([colour, histogram])

        pygame.quit()
