import argparse, pygame, time, math, random, re, json, copy, os
from PIL import Image
from os import path

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GL import shaders


from scipy.stats import ortho_group
from scipy.linalg import logm, expm


from fractalflames3d import Iterator, Renderer, genTexture3D, fix_image_paths, rotationY_mat

def lerp_matrix(a, b, t):
    return expm(logm(a)*(1-t) + logm(b)*t)


def polar_decomposition(mat):
    u, s, vh = np.linalg.svd(mat)

    p = (vh.conj().T * s) @ vh
    u = u @ vh

    return u, p


'''mat_a = np.array([[math.cos(0.2), -math.sin(0.2)], [math.sin(0.2), math.cos(0.2)]])# * 2
mat_b = np.array([[math.cos(1), -math.sin(1)], [math.sin(1), math.cos(1)]]) 

lerped = lerp_matrix(mat_a, mat_b, 0.5)

print(np.linalg.det(mat_a), np.linalg.det(mat_b), np.linalg.det(lerped))

print(lerped)

assert False'''

def generate_next_matrix():
    current = np.empty((3,4), float)

    transform = current[:,:-1]

    offset = current[:,-1]
    offset[:] = np.random.standard_normal(3) 
    offset /= np.sqrt(np.sum(offset**2))

    transform[:,:] = ortho_group.rvs(3) + np.random.normal(0, 0.2, (3,3))

    det = np.linalg.det(transform)

    if det < 0:
        transform *= -1

    transform *= (0.9 / abs(det))**(1/3)

    
    return current


def lerp(flame_a, flame_b, t):
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

            #transform[:,:] = ortho_group.rvs(3) + np.random.normal(0, 0.2, (3,3))
            #transform *= (0.9 / abs(np.linalg.det(transform)))**(1/3)


            #print(np.linalg.det(mat_a[:,:-1]), np.linalg.det(mat_b[:,:-1]), np.linalg.det(res[:,:-1]))

            func[key] = (res).tolist()



        
        result['functions'].append(func)

    '''print(flame_a)
    print(flame_b)

    print()
    print(result)
    assert False'''

    return result

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

    cycles = 140#70

    keys = ((K_d, K_a),(K_q,K_e),(K_s, K_w),(K_KP8,K_KP2),(K_KP4,K_KP6),(K_KP7,K_KP9),(K_LCTRL,K_LSHIFT))

    pos = np.array(flame['pos'], float)
    rot = np.array(flame['rot'], float)

    flames = [copy.deepcopy(flame)]
    i = 0

    animate = False

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
                    elif event.key == K_t:
                        animate = not animate
                    elif event.key == K_i:
                        i = i + 1

                        while i >= len(flames):
                            new_flame = copy.deepcopy(flame)
                            functions = []
                            for func in flame['functions']:
                                func = func.copy()

                                func['pre_trans'] = generate_next_matrix().tolist()
                                func['post_trans'] = generate_next_matrix().tolist()

                                functions.append(func)
                            new_flame['functions'] = functions

                            flames.append(new_flame)


                            #lerp(flames[0], flames[1], 0.5)
                    elif event.key == K_k:
                        prev_i = i
                        i = (i - 1) % len(flames)
                    elif event.key == K_p:
                        print(json.dumps(flames[i]))

            pressed = pygame.key.get_pressed()

            action = np.array([pressed[b] - pressed[a] for a, b in keys])
            rot += action[3:6] / 30

            speed = [0.03, 0.1, 0.3][action[6]+1]
            pos -= action[0:3] @ rotationY_mat(-rot[1])[:3,:3] * speed

            pygame.display.set_caption('{}: {:.2f} ({:,}) - [{:.3f}, {:.3f}, {:.3f}] - {}'.format(title, clock.get_fps(), round(clock.get_fps() * iterator.iterations_per_tick * cycles), *pos, i))
            glClear(GL_COLOR_BUFFER_BIT)

            if animate:
                flame = lerp(flame, flames[i], 0.02)
            else:
                flame = flames[i]

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