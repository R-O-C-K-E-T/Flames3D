import argparse, pygame, time, math, random, re, json, copy, os, shutil
from PIL import Image
from os import path

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

from scipy.stats import ortho_group
from scipy.linalg import logm, expm

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GL import shaders

from fractalflames3d import Iterator, Renderer, genTexture3D, fix_image_paths, rotationY_mat
from render import FileRenderer

def smoothstep(t):
    return t*t*(3 - 2*t)

def lerp_matrix(a, b, t):
    return expm(logm(a)*(1-t) + logm(b)*t)

def polar_decomposition(mat):
    u, s, vh = np.linalg.svd(mat)

    p = (vh.conj().T * s) @ vh
    u = u @ vh

    return u, p

def generate_random_matrix():
    current = np.empty((3,4), float)

    det = 0.9
    length = 1.0

    transform = current[:,:-1]
    offset = current[:,-1]

    offset[:] = np.random.standard_normal(3) 
    offset *= length / np.sqrt(np.sum(offset**2))

    transform[:,:] = ortho_group.rvs(3) + np.random.normal(0, 0.2, (3,3))

    current_det = np.linalg.det(transform)
    if current_det < 0:
        transform *= -1

    transform *= (det / abs(current_det))**(1/3)

    
    return current


def lerp_flame(flame_a, flame_b, t):
    #t = 0.5
    result = flame_a.copy()
    result['functions'] = []

    for func_a, func_b in zip(flame_a['functions'], flame_b['functions']):
        func = func_a.copy()

        for key in ('pre_trans', 'post_trans'):
            mat_a = np.array(func_a[key])
            mat_b = np.array(func_b[key])
            res = mat_a * (1-t) + mat_b * t

            transform = res[:,:-1]
            transform[:,:] = lerp_matrix(mat_a[:,:-1], mat_b[:,:-1], t)

            func[key] = res.tolist()
        
        result['functions'].append(func)

    '''print(flame_a)
    print(flame_b)

    print()
    print(result)
    assert False'''

    return result

def zero_pad(digits, n):
    v = str(n)
    while len(v) < digits:
        v = '0' + v
    return v

def generate_random_flame(source_flame):
    flame = copy.deepcopy(source_flame)

    for func in flame['functions']:
        func['pre_trans'] = generate_random_matrix().tolist()
        func['post_trans'] = generate_random_matrix().tolist()

    return flame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a 3d fractal flame')

    parser.add_argument('flame', type=str, help='Base flame to animate')
    parser.add_argument('count', type=int, help='Number of random flames to create')

    parser.add_argument('--fast', '-f', action='store_true', help='Flag to switch to low quality mode')
    parser.add_argument('--duration', '-d', type=float, default=20, help='Duration of output video')


    args = parser.parse_args()

    if args.count < 2:
        raise RuntimeError('Count < 2')

    if args.fast:
        cycles = 500
        width = 500
        height = 500
        depth = 64
        framerate = 15
    else:
        cycles = 2000
        width = 1000
        height = 1000
        depth = 256
        framerate = 30

    duration = args.duration
    if duration <= 0:
        raise RuntimeError('Duration <= 0')

    frames = max(round(framerate * duration), 1)
    digits = math.ceil(math.log(frames, 10))

    original_flame = json.load(open(args.flame))
    fix_image_paths(original_flame, path.dirname(args.flame))

    size = width, height, depth
    pygame.display.set_mode((1,1), OPENGL)

    glClearColor(0,0,0,1)

    colour = genTexture3D()
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, *size, 0, GL_RGBA, GL_FLOAT, None)
    histogram = genTexture3D()
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, *size, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, None)

    try:
        iterator = Iterator(histogram, colour)
        renderer = Renderer(histogram, colour)
        file_renderer = FileRenderer(size[:2])

        dest_folder = 'output'
        shutil.rmtree(dest_folder)
        os.makedirs(dest_folder)

        flames = []
        while len(flames) < args.count:
            flame = generate_random_flame(original_flame)

            iterator.set_flame(flame)
            positions = iterator.dump_particle_positions()
            centre = positions.mean(0)
            if (abs(centre) < 1).all():
                flames.append((flame, centre))

            iterator.reset_particles()

        print()
        for i in range(frames):
            print('\r{}/{}'.format(zero_pad(digits, i), frames), end='')
            v = i / frames

            flame_a, centre_a = flames[math.floor(v * len(flames))]
            flame_b, centre_b = flames[math.floor(v * len(flames) + 1) % len(flames)]

            t = smoothstep((v * len(flames)) % 1)

            flame = lerp_flame(flame_a, flame_b, t)
            centre = centre_a*(1-t) + centre_b*t

            iterator.set_flame(flame)

            #positions = iterator.dump_particle_positions()
            #centre = positions.mean(0)

            a = 2*math.pi * 5 * v 
            flame = flame.copy()
            flame['pos'] = (centre + [6*math.sin(a), 0, 6*math.cos(a)]).tolist()
            flame['rot'] = [0, -a, 0]

            iterator.set_flame(flame)
            for _ in range(cycles):
                iterator.cycle()

            with file_renderer(path.join(dest_folder, '{}.png'.format(zero_pad(digits, i)))):
                renderer.render(flame, iterator.tick * iterator.iterations_per_tick)
        print('\r{}/{}'.format(frames, frames))
        try:
            os.remove('out.mp4')
        except FileNotFoundError:
            pass
        os.system('ffmpeg -framerate {} -pattern_type glob -i \'{}/*.png\' -c:v libx264 -pix_fmt yuv420p out.mp4'.format(framerate, dest_folder))
    finally:
        iterator.cleanup()
        renderer.cleanup()
        file_renderer.cleanup()
        glDeleteTextures([colour, histogram])

        pygame.quit()

