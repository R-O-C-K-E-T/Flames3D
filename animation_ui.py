import argparse, pygame, time, math, random, re, json, threading, queue, string, copy
import tkinter as tk
import numpy as np

from traceback import print_exc
from os import path

from ctypes import *

from pygame.locals import *

from OpenGL.GL import *

from tkinter import filedialog

from idlelib.percolator import Percolator
from idlelib.colorizer import ColorDelegator
from idlelib.statusbar import MultiStatusBar

from fractalflames3d import rotationY_mat, Iterator, Renderer, genTexture3D, fix_image_paths


from fractalflames3d import rotation_mat, projection_mat, translation_mat

class TextWindow(tk.Frame):
    def __init__(self, master, initial='', width=90, **kwargs):
        super().__init__(master, **kwargs)

        self.text = tk.Text(self, name='text', padx=5, wrap='none', width=width, undo=True)
        vbar = tk.Scrollbar(self, name='vbar')
        vbar['command'] = self.text.yview
        vbar.pack(side=tk.LEFT, fill=tk.Y)
        self.text['yscrollcommand'] = vbar.set

        self.status_bar = MultiStatusBar(self)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.set_content(initial)
        self.text.edit_reset()
        self.text.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        Percolator(self.text).insertfilter(ColorDelegator())

        self.text.bind("<<set-line-and-column>>", self.set_line_and_column)
        self.text.event_add("<<set-line-and-column>>",
                            "<KeyRelease>", "<ButtonRelease>")
        self.text.after_idle(self.set_line_and_column)

        self.text.event_add("<<set-line-and-column>>",
                            "<KeyRelease>", "<ButtonRelease>")

        self.listeners = []


    IDENTCHARS = string.ascii_letters + string.digits + "_"
    def check_syntax(self):
        try:
            return compile(self.get_content(), '<string>', 'exec')
        except (SyntaxError, OverflowError, ValueError) as value:
            msg = getattr(value, 'msg', '') or value or "<no detail available>"
            lineno = getattr(value, 'lineno', '') or 1
            offset = getattr(value, 'offset', '') or 0
            if offset == 0:
                lineno += 1  #mark end of offending line
            pos = "0.0 + %d lines + %d chars" % (lineno-1, offset-1)

            self.text.tag_add("ERROR", pos)
            char = self.text.get(pos)
            if char and char in self.IDENTCHARS:
                self.text.tag_add("ERROR", pos + " wordstart", pos)
            if '\n' == self.text.get(pos):   # error at line end
                self.text.mark_set("insert", pos)
            else:
                self.text.mark_set("insert", pos + "+1c")
            self.text.see(pos)

            return msg

    def add_content_change_listener(self, callback):
        self.listeners.append(callback)

    def set_line_and_column(self, event=None):
        line, column = self.text.index(tk.INSERT).split('.')
        self.status_bar.set_label('column', 'Col: %s' % column, side=tk.RIGHT)
        self.status_bar.set_label('line', 'Ln: %s' % line, side=tk.RIGHT)


        for callback in self.listeners:
            callback()

    def get_content(self):
        return self.text.get("1.0",tk.END)

    def set_content(self, text):
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, text)

def run_ui_thread(event_queue):
    global root
    root = tk.Tk()
    root.title('Animation')

    inner = tk.Frame(root)
    inner.pack(padx=5, pady=5)

    editor = TextWindow(inner)
    editor.pack()

    def content_change(*args):
        try:
            code = compile(editor.get_content(), '<string>', 'exec')
        except:
            return
        event_queue.put(('code', code))    
    #editor.add_content_change_listener(content_change)

    error = tk.StringVar(inner)

    def submit():
        result = editor.check_syntax()
        if isinstance(result, str):
            error.set(result)
            return
        error.set('')
        event_queue.put(('code', result))

    tk.Label(inner, textvariable=error).pack()

    footer = tk.Frame(inner)
    footer.pack()

    frames_var = tk.IntVar(inner, value=600)

    def update_frames(*_):
        event_queue.put(('frames', frames_var.get()))
    frames_var.trace('w', update_frames)
    tk.Spinbox(footer, textvariable=frames_var, from_=0, to=1000, increment=1, repeatdelay=50, width=5).pack(side=tk.LEFT)

    tk.Frame(footer, width=20).pack(side=tk.LEFT)

    tk.Button(footer, command=submit, text='Submit').pack(side=tk.LEFT)

    tk.Frame(footer, width=20).pack(side=tk.LEFT)

    def save():
        filename = filedialog.asksaveasfilename(title='Save Animation Script', defaultextension=".py", filetypes=(
        ('Python files', '*.py'), ('All files', '*.*')))
        if filename == '' or isinstance(filename, tuple):
            return
        with open(filename, 'w') as f:
            f.write(editor.get_content())    
    
    tk.Button(footer, command=save, text='Save').pack(side=tk.LEFT)
    def load():
        filename = filedialog.askopenfilename(title='Load Animation Script', filetypes=(('Python files', '*.py'), ('All files', '*.*')))
        if filename == '' or isinstance(filename, tuple):
            return
        with open(filename) as f:
            editor.set_content(f.read())
        
        
    tk.Button(footer, command=load, text='Load').pack(side=tk.LEFT)

    def on_closing(a):
        if a.widget != root:
            return
        event_queue.put(('exit', None))
    root.bind("<Destroy>", on_closing)

    root.mainloop()


def get_ndc():
    points = []
    for a in (-1, 1):
        for b in (-1, 1):
            points.append([-1,  a,  b])
            points.append([+1,  a,  b])

            points.append([ a, -1,  b])
            points.append([ a, +1,  b])

            points.append([ a,  b, -1])
            points.append([ a,  b, +1])

    return np.array(points, dtype=float)

def draw_ndc():
    glBegin(GL_LINES)
    for a in (-1, 1):
        for b in (-1, 1):
            glVertex3f(-1, a, b)
            glVertex3f(+1, a, b)

            glVertex3f(a, -1, b)
            glVertex3f(a, +1, b)
            
            glVertex3f(a, b, -1)
            glVertex3f(a, b, +1)
    glEnd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a 3d fractal flame')

    parser.add_argument('flame', type=str)

    parser.add_argument("--width", type=int, default=1000, help='Image width')
    parser.add_argument("--height", type=int, default=1000, help='Image height')
    parser.add_argument("--depth", type=int, default=64, help='Depth Resolution')


    args = parser.parse_args()

    original_flame = json.load(open(args.flame))
    display = args.width, args.height
    pygame.display.set_mode(display, OPENGL|DOUBLEBUF)

    glViewport(0, 0, *display)
    glEnable(GL_FRAMEBUFFER_SRGB)

    glPointSize(10)

    glClearColor(0,0,0,1)

    clock = pygame.time.Clock()

    title = 'Fractal Flames 3D'
    pygame.display.set_caption(title)

    event_queue = queue.Queue()

    ui_thread = threading.Thread(target=run_ui_thread, name='UI Thread', args=(event_queue,))
    ui_thread.start()

    size = *display, args.depth

    colour = genTexture3D()
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, *size, 0, GL_RGBA, GL_FLOAT, None)
    histogram = genTexture3D()
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, *size, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, None)

    iterator = Iterator(histogram, colour)
    renderer = Renderer(histogram, colour)

    cycles = 70

    keys = ((K_d, K_a),(K_q,K_e),(K_s, K_w),(K_KP8,K_KP2),(K_KP4,K_KP6),(K_KP7,K_KP9),(K_LCTRL,K_LSHIFT))

    pos = np.array(original_flame['pos'], float)
    rot = np.array(original_flame['rot'], float)

    frames = 600

    code = None
    
    time = 0
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pass#root.destroy()
                elif event.type == KEYDOWN:
                    if event.key == K_r:
                        original_flame = json.load(open(args.flame))
                        iterator.reload_images()
            try:
                while True:
                    name, data = event_queue.get_nowait()

                    if name == 'exit':
                        running = False
                    elif name == 'code':
                        code = data
                    elif name == 'frames':
                        frames = max(data, 1)
                    else:
                        assert False
            except queue.Empty:
                pass

            flame = copy.deepcopy(original_flame)
            pressed = pygame.key.get_pressed()

            action = np.array([pressed[b] - pressed[a] for a, b in keys])
            rot += action[3:6] / 30

            speed = [0.03, 0.1, 0.3][action[6]+1]
            pos -= action[0:3] @ rotationY_mat(-rot[1])[:3,:3] * speed

            pygame.display.set_caption('{}: {:.2f} ({:,}) - [{:.3f}, {:.3f}, {:.3f}]'.format(title, clock.get_fps(), round(clock.get_fps() * iterator.iterations_per_tick * cycles), *pos))
            glClear(GL_COLOR_BUFFER_BIT)

            flame['pos'] = pos.tolist()
            flame['rot'] = rot.tolist()

            if code is not None:
                for func in flame['functions']:
                    func['pre_trans'] = np.array(func['pre_trans'])
                    func['post_trans'] = np.array(func['post_trans'])

                t = (time % frames) / frames
                try:
                    exec(code, globals(), {})
                except:
                    print_exc()
                    code = None
                
                for func in flame['functions']:
                    func['pre_trans'] = func['pre_trans'].tolist()
                    func['post_trans'] = func['post_trans'].tolist()

            fix_image_paths(flame, path.dirname(args.flame))

            iterator.set_flame(flame)

            for _ in range(cycles):
                iterator.cycle()

            renderer.render(flame, iterator.tick * iterator.iterations_per_tick)
            pygame.display.flip()
            clock.tick(60)
            time += 1
    finally:
        #print(pos.tolist(), rot.tolist())

        #glBindTexture(GL_TEXTURE_3D, colour)
        #colour_data = glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, np.empty((*size, 4), np.float32))

        if False:
            glBindTexture(GL_TEXTURE_3D, histogram)
            histogram_data = glGetTexImage(GL_TEXTURE_3D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, np.empty(size[::-1], np.uint32))
            np.save('temp', histogram_data)

        iterator.cleanup()
        renderer.cleanup()

        glDeleteTextures([colour, histogram])

        pygame.quit()
