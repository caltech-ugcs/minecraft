from sdl2 import *
from OpenGL import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np

# OpenGL follows a pipeline computation model in which arrays of vertices are 
# rendered using a shader program. A vertex array object (vao) stores the attributes
# of all vertices in a 3D model (position, color, etc). The vertex shader program uses
# these attributes to compute a 2D projection of the model. The fragment shader computes
# the color of each pixel in the rasterized 2D projection of the model.
program = None
vao = None

def create_shader_program():
    # Apply transformation to each vertex using GL shader language
    # GLSL and assign the resulting position to the gl_Position global
    # variable. In this case, we apply the identity transformation to
    # each vertex position.
    vertexShader = shaders.compileShader("""
    #version 410
    layout (location=0) in vec3 position;
    void main()
    {
        gl_Position = vec4(position, 1.0);
    }
    """, GL_VERTEX_SHADER)

    # Set the color of each pixel to some GRBA value. In this case,
    # we set the color of each pixel of each triangle to white. 
    fragmentShader = shaders.compileShader("""
    #version 410
    out vec4 color;
    void main()
    {
        color = vec4(1.0, 1.0, 1.0, 1.0);
    }
    """, GL_FRAGMENT_SHADER)

    return shaders.compileProgram(vertexShader, fragmentShader)

def init():
    """
    Initialize the global variables containing the vertices of a single triangle and
    a compiled OpenGL shader. 
    """
    global program
    global vao

    # Create an array containing the positions of the vertices of a single triangle. Note that 
    # the vertices must be specified in counter-
    vertex_position = np.array([
        -0.5, -0.5, 0.0, # vertex 1
        0.5, -0.5, 0.0,  # vertex 2
        0.0,  0.5, 0.0   # vertex 3
    ], dtype=np.float32)

    # Create a vertex array object (vao) that will contain a vertex buffer object (vbo) for
    # all vertex attributes.
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Create a vertex buffer object (vbo) containing the position attribute for each vertex
    # of the model. Load the position data into GPU memory with GL_STATIC_DRAW storage class.
    # This indicates that the vertex positions will not be updated frequently.
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_position.nbytes, vertex_position, GL_STATIC_DRAW)

    # Assign the currently bound vertex buffer object (vbo) to the vertex attribute variable at
    # location 0 and specify its type (vec3).
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    program = create_shader_program()

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

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
    global vao

    # Clear the screen with black
    glClearColor(0, 0, 0, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Enable the shader program and vertex-array-object (vao)
    glUseProgram(program)
    glBindVertexArray(vao)

    # Draw 3 vertices from the enabled vertex-array-object (vao) using 
    # the enabled shader program.
    glDrawArrays(GL_TRIANGLES, 0, 3)

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
    # Create a SDL2 window that is drawn with a OpenGL 4.1 context. 
    # This is the most recent version of OpenGL supported by macOS.
    SDL_Init(SDL_INIT_VIDEO)
    window = SDL_CreateWindow(b"Tutorial 1", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 600, 600, SDL_WINDOW_OPENGL)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
    ctx = SDL_GL_CreateContext(window)

    # Perform initial setup (load geometry, compile shaders)
    init()
    
    # Render animation until window is closed.
    run(window)

    # Destroy the OpenGL context and the SDL2 window.
    SDL_GL_DeleteContext(ctx)
    SDL_DestroyWindow(window)
    SDL_Quit()
    return 0

if __name__ == "__main__":
    main()