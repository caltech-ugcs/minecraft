from sdl2 import *
from OpenGL import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import glm
import time
from minecraft.Chunk import Chunk
from minecraft.World import World

# OpenGL follows a pipeline computation model in which arrays of vertices are 
# rendered using a shader program. A vertex array object (vao) stores the attributes
# of all vertices in a 3D model (position, color, etc). The vertex shader program uses
# these attributes to compute a 2D projection of the model. The fragment shader computes
# the color of each pixel in the rasterized 2D projection of the model.
program = None
world = None

def set_mat4(name: str, mat: glm.mat4) -> None:
    glUniformMatrix4fv(glGetUniformLocation(program, name), 1, GL_FALSE, glm.value_ptr(mat))

def create_shader_program():
    # Apply transformation to each vertex using GL shader language
    # GLSL and assign the resulting position to the gl_Position global
    # variable. In this case, we apply the identity transformation to
    # each vertex position.
    vertexShader = shaders.compileShader("""
    #version 410
    layout (location=0) in vec3 x;
    layout (location=1) in vec3 n;
    uniform mat4 model_x;
    uniform mat4 model_n;
    uniform mat4 view;
    uniform mat4 projection;
    out vec3 v_color;

    vec3 lighting(vec3 v, vec3 n) {
        float ambient = 0.2;
        float diffuse = max(0.0, dot(n, normalize(vec3(0.25, 0.75, 0.5))));
        float intensity = min(1.0, ambient + diffuse);
        return vec3(1.0, 1.0, 1.0) * intensity;
    }

    void main() {
        gl_Position = projection * view * model_x * vec4(x, 1.0);
        v_color = lighting(x, (model_n * vec4(n, 1)).xyz);
    }
    """, GL_VERTEX_SHADER)

    # Set the color of each pixel to some GRBA value. In this case,
    # we set the color of each pixel of each triangle to white. 
    fragmentShader = shaders.compileShader("""
    #version 410
    in vec3 v_color;
    out vec4 color;

    void main() {
        color = vec4(v_color, 1.0);
    }
    """, GL_FRAGMENT_SHADER)

    return shaders.compileProgram(vertexShader, fragmentShader)

def render():
    """
    Draw a single frame of the animation using OpenGL accelerated functions. 
    1. Clear the screen.
    2. Specify the shader to use.
    3. Specify the vertices to draw.
    4. Draw the vertices.
    5. Cleanup
    """

    global program
    global world

    # Clear the screen with black
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    for chunk in world.chunks.values():

        # Enable the shader program and vertex-array-object (vao)
        glUseProgram(program)
        glBindVertexArray(chunk.vao)

        # translate model so it is centered at origin
        model_x = chunk.model_x
        # rotate model around origin at 22.5 degrees per second
        angle = (time.time() * 22.5) % 360
        q = glm.quat(glm.vec3(0, glm.radians(angle), 0))
        model_x = glm.mat4(q) * model_x
        set_mat4("model_x", model_x)

        model_n = glm.inverse(glm.transpose(model_x))
        set_mat4("model_n", model_n)

        # Rotate camera downwards
        q = glm.quat(glm.vec3(glm.radians(-22.5), 0, 0))
        # Move camera up and back to view ground
        view = glm.translate(glm.vec3(0, 136, 24)) * glm.mat4(q)
        view = glm.inverse(view)
        set_mat4("view", view)

        projection = glm.perspective(45.0, 1.0, 0.1, 100.0)
        set_mat4("projection", projection)

        # Draw 3 vertices from the enabled vertex-array-object (vao) using 
        # the enabled shader program.
        glDrawArrays(GL_TRIANGLES, 0, chunk.n_vertices)

        # Disable the shader program and vertex-array-object (vao)
        glBindVertexArray(0)
        glUseProgram(0)

  
def run(window):
    """
    Main event loop. Renders a single triangle using OpenGL until the SDL2 window
    is closed. This function should be extended in the future if mouse or keyboard
    events should be handled
    """
    import ctypes
    event = SDL_Event()
    running = True
    while running:
        while SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == SDL_QUIT:
                running = False
        render()
        SDL_GL_SwapWindow(window)


def main():
    global world
    global program

    # Create a SDL2 window that is drawn with a OpenGL 4.1 context. 
    # This is the most recent version of OpenGL supported by macOS.
    SDL_Init(SDL_INIT_VIDEO)
    window = SDL_CreateWindow(b"Tutorial 2", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 600, 600, SDL_WINDOW_OPENGL)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
    ctx = SDL_GL_CreateContext(window)

    # Perform initial setup (load geometry, compile shaders)
    world = World(420)
    for u in range (-1, 2):
        for v in range (-1, 2):
            world.generate(u, v)
    world.update()
    
    glBindVertexArray(world.chunks[(0, 0)].vao)
    program = create_shader_program()
    
    # Render animation until window is closed.
    run(window)

    # Destroy the OpenGL context and the SDL2 window.
    SDL_GL_DeleteContext(ctx)
    SDL_DestroyWindow(window)
    SDL_Quit()
    return 0

if __name__ == "__main__":
    main()