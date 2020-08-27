#!/usr/bin/python3
import argparse, pygame, time, os, traceback, json, math

from contextlib import contextmanager
from OpenGL.GL import *

from PIL import Image
from os import path

from pygame.locals import OPENGL

from fractalflames3d import Iterator, Renderer, genTexture3D, fix_image_paths

class FileRenderer:
    def __init__(self, size):
        self.size = size

        self.framebuffer = int(glGenFramebuffers(1))
        self.texture = int(glGenTextures(1))

        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)

        glEnable(GL_FRAMEBUFFER_SRGB)

        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8, *size, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.texture, 0)

    def cleanup(self):
        if self.framebuffer is not None:
            glDeleteFramebuffers(1, [self.framebuffer])
            self.framebuffer = None

        if self.texture is not None:
            glDeleteTextures([self.texture])
            self.texture = None

    @contextmanager
    def __call__(self, file):
        prev_viewport = glGetIntegerv(GL_VIEWPORT)

        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
        glBindTexture(GL_TEXTURE_2D, self.texture)


        glViewport(0, 0, *self.size)
        try:
            yield None
        finally:
            glMemoryBarrier(GL_ALL_BARRIER_BITS) # Dunno man

            glBindTexture(GL_TEXTURE_2D, self.texture)
            pixel_data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)

            img = Image.frombytes('RGB', self.size, pixel_data)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if isinstance(file, str):
                with open(file, 'wb') as f:
                    img.save(f)
            else:
                img.save(file)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(*prev_viewport)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a 3d fractal flame')

    parser.add_argument("flames", type=argparse.FileType('r'), nargs='+', help='Flame(s) to render')

    parser.add_argument("--dest", type=str, default='.', help='Output image directory')
    parser.add_argument("--width", type=int, default=1000, help='Image width')
    parser.add_argument("--height", type=int, default=1000, help='Image height')
    parser.add_argument("--depth", type=int, default=64, help='Depth Resolution')
    parser.add_argument("--cycles", type=int, default=1_000, help='Number of cycles to perform')

    args = parser.parse_args()

    if args.width <= 0:
        raise RuntimeError("width <= 0")
    if args.height <= 0:
        raise RuntimeError("height <= 0")
    if args.depth <= 0:
        raise RuntimeError("depth <= 0")
    if args.cycles <= 0:
        raise RuntimeError("cycles <= 0")

    completed = 0

    initial = time.time()
    pygame.display.set_mode((1,1), OPENGL)

    try:
        size = args.width, args.height, args.depth

        colour = genTexture3D()
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, *size, 0, GL_RGBA, GL_FLOAT, None)
        histogram = genTexture3D()
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, *size, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, None)

        iterator = Iterator(histogram, colour)
        renderer = Renderer(histogram, colour)

        file_renderer = FileRenderer(size[:2])

        for in_file in args.flames:
            t = time.time()
            try:
                flame = json.load(in_file)
                fix_image_paths(flame, path.dirname(in_file.name))
                iterator.set_flame(flame)
            except:
                print('Failed to interpret:', in_file.name)
                traceback.print_exc()
                continue

            dest_name = path.join(args.dest, path.splitext(path.basename(in_file.name))[0] + '.png')
            try:
                os.makedirs(path.dirname(dest_name))
            except FileExistsError:
                pass

            try:
                dest_file = open(dest_name, 'wb')
            except:
                print('Failed to open:', dest_name)
                continue
            #print(iterator.program)

            max_iterations = args.cycles
            BLOCK_SIZE = 1000
            for _ in range(max_iterations // BLOCK_SIZE):
                for _ in range(BLOCK_SIZE):
                    iterator.cycle()
                glFinish()
            for _ in range(max_iterations % BLOCK_SIZE):
                iterator.cycle()

            with file_renderer(dest_file):
                renderer.render(flame, iterator.tick * iterator.iterations_per_tick)

            dest_file.close()

            dt = time.time() - t
            completed += 1
            print('Rendered {} to {} in {:.2f}s @ {:,} iterations per second'.format(in_file.name, dest_name, dt, round(args.cycles * iterator.iterations_per_tick / dt)))
        print('{}/{} rendered in {:.2f}s'.format(completed, len(args.flames), time.time() - initial))
    finally:
        iterator.cleanup()
        renderer.cleanup()
        file_renderer.cleanup()
        glDeleteTextures([colour, histogram])

        pygame.quit()
