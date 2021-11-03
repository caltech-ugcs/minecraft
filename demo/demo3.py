from sdl2 import *
from OpenGL import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import glm
import time
from perlin_noise.perlin_noise import PerlinNoise

FACE_LEFT = 0
FACE_BOTTOM = 1
FACE_FRONT = 2
FACE_RIGHT = 3
FACE_TOP = 4
FACE_BACK = 5

cube_vertices = [
    [
    -0.5,-0.5,-0.5, -0.5,-0.5, 0.5, -0.5, 0.5, 0.5,
    -0.5,-0.5,-0.5, -0.5, 0.5, 0.5, -0.5, 0.5,-0.5,
    ],
    [
     0.5,-0.5, 0.5, -0.5,-0.5, 0.5, -0.5,-0.5,-0.5,
     0.5,-0.5, 0.5, -0.5,-0.5,-0.5,  0.5,-0.5,-0.5,
    ],
    [    
     0.5, 0.5,-0.5, -0.5,-0.5,-0.5, -0.5, 0.5,-0.5,
     0.5, 0.5,-0.5,  0.5,-0.5,-0.5, -0.5,-0.5,-0.5,
    ],
    [   
     0.5, 0.5, 0.5,  0.5,-0.5,-0.5,  0.5, 0.5,-0.5,
     0.5,-0.5,-0.5,  0.5, 0.5, 0.5,  0.5,-0.5, 0.5,
    ],
    [
     0.5, 0.5, 0.5,  0.5, 0.5,-0.5, -0.5, 0.5,-0.5,
     0.5, 0.5, 0.5, -0.5, 0.5,-0.5, -0.5, 0.5, 0.5,
    ],
    [    
     0.5, 0.5, 0.5, -0.5, 0.5, 0.5,  0.5,-0.5, 0.5,
    -0.5, 0.5, 0.5, -0.5,-0.5, 0.5,  0.5,-0.5, 0.5,
    ]
]

cube_normals = [
    [
    -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
    -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
    ],
    [
     0.0,-1.0, 0.0,  0.0,-1.0, 0.0,  0.0,-1.0, 0.0,
     0.0,-1.0, 0.0,  0.0,-1.0, 0.0,  0.0,-1.0, 0.0,
    ],
    [    
     0.0, 0.0,-1.0,  0.0, 0.0,-1.0,  0.0, 0.0,-1.0,
     0.0, 0.0,-1.0,  0.0, 0.0,-1.0,  0.0, 0.0,-1.0,
    ],
    [   
     1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
     1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 0.0, 0.0,
    ],
    [
     0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0,
     0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0,
    ],
    [    
     0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,
     0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 1.0,
    ]
]

def generate_chunk(seed):
    """
    Generate a single chunk with the given seed according to the following algorithm:
    For each (x, z) coordinate, compute the perlin noise function. Fill in all blocks
    with a height less than ground + amplitude * noise(x, z).
    """
    noise = PerlinNoise(seed=seed)
    chunk = np.zeros((16, 256, 16), dtype=np.int16)
    amplitude = 10
    ground = 128
    for x in range(0, 16):
        for z in range(0, 16):
            height = int(ground + amplitude * noise([x / 16, z / 16]))
            for y in range(0, 256):
                if y < height:
                    chunk[x,y,z] = 1
    return chunk

def translate(vertices, x, y, z):
    """
    Translate the vertex buffer by the given x, y, and z offsets
    and return the result.
    """
    n = len(vertices)
    vertices = np.reshape(vertices, (n // 3, 3))
    vertices[:,0] += x
    vertices[:,1] += y
    vertices[:,2] += z
    return np.reshape(vertices, (n)).tolist()

def compute_mesh(chunk):
    """
    Compute a mesh containing only visible faces using the following algorithm:
    For all blocks b that are not air:
        For all faces f of b:
            If the block adjacent to f is not air:
                Add the face to the mesh
    
    Note that faces are easily computed by translating the faces of the unit
    cube using the translate function defined above.
    """
    vertex_x = []
    vertex_n = []
    for x in range(0, 16):
        for z in range(0, 16):
            for y in range(0, 256):
                if chunk[x,y,z] == 0.0:
                    continue
                if x < 15 and chunk[x + 1, y, z] == 0:
                    vertex_x += translate(cube_vertices[FACE_RIGHT], x, y, z)
                    vertex_n += cube_normals[FACE_RIGHT]
                if x > 0 and chunk[x - 1, y, z] == 0:
                    vertex_x += translate(cube_vertices[FACE_LEFT], x, y, z)
                    vertex_n += cube_normals[FACE_LEFT]
                if y < 255 and chunk[x, y + 1, z] == 0:
                    vertex_x += translate(cube_vertices[FACE_TOP], x, y, z)
                    vertex_n += cube_normals[FACE_TOP]
                if y > 0 and chunk[x, y - 1, z] == 0:
                    vertex_x += translate(cube_vertices[FACE_BOTTOM], x, y, z)
                    vertex_n += cube_normals[FACE_BOTTOM]
                if z < 15 and chunk[x, y, z + 1] == 0:
                    vertex_x += translate(cube_vertices[FACE_BACK], x, y, z)
                    vertex_n += cube_normals[FACE_BACK]
                if z > 0 and chunk[x, y, z - 1] == 0:
                    vertex_x += translate(cube_vertices[FACE_FRONT], x, y, z)
                    vertex_n += cube_normals[FACE_FRONT]
    return (np.array(vertex_x, dtype=np.float32), np.array(vertex_n, dtype=np.float32))

chunk = generate_chunk(420)
mesh_x, mesh_n = compute_mesh(chunk)
print(mesh_x)

# OpenGL follows a pipeline computation model in which arrays of vertices are 
# rendered using a shader program. A vertex array object (vao) stores the attributes
# of all vertices in a 3D model (position, color, etc). The vertex shader program uses
# these attributes to compute a 2D projection of the model. The fragment shader computes
# the color of each pixel in the rasterized 2D projection of the model.
program = None
vao = None

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

def init():
    """
    Initialize the global variables containing the vertices of a single triangle and
    a compiled OpenGL shader. 
    """
    global program
    global vao

    # Create a vertex array object (vao) that will contain a vertex buffer object (vbo) for
    # all vertex attributes.
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Create a vertex buffer object (vbo) containing the position attribute for each vertex
    # of the model. Load the position data into GPU memory with GL_STATIC_DRAW storage class.
    # This indicates that the vertex positions will not be updated frequently.
    vbo_position = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_position)
    glBufferData(GL_ARRAY_BUFFER, mesh_x.nbytes, mesh_x, GL_STATIC_DRAW)

    # Assign the currently bound vertex buffer object (vbo) to the vertex attribute variable at
    # location 0 and specify its type (vec3).
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Create a vertex buffer object (vbo) containing the normal attribute for each vertex
    # of the model. Load the normal data into GPU memory with GL_STATIC_DRAW storage class.
    # This indicates that the vertex normals will not be updated frequently.
    vbo_normal = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_normal)
    glBufferData(GL_ARRAY_BUFFER, mesh_n.nbytes, mesh_n, GL_STATIC_DRAW)

    # Assign the currently bound vertex buffer object (vbo) to the vertex attribute variable at
    # location 0 and specify its type (vec3).
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

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
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    # Enable the shader program and vertex-array-object (vao)
    glUseProgram(program)
    glBindVertexArray(vao)

    # translate model so it is centered at origin
    model_x =  glm.translate(glm.vec3(-8, 0, -8))
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
    glDrawArrays(GL_TRIANGLES, 0, len(mesh_x) // 3)

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
    window = SDL_CreateWindow(b"Tutorial 2", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 600, 600, SDL_WINDOW_OPENGL)
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