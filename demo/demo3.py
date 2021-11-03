import numpy as np
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
    chunk = np.zeros((16, 256, 16))
    amplitude = 20
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
                if x < 15 and chunk[x + 1, y, z] == 0.0:
                    vertex_x += translate(cube_vertices[FACE_RIGHT], x, y, z)
                    vertex_n += cube_normals[FACE_RIGHT]
                if x > 0 and chunk[x - 1, y, z] == 0.0:
                    vertex_x += translate(cube_vertices[FACE_LEFT], x, y, z)
                    vertex_n += cube_normals[FACE_LEFT]
                if y < 255 and chunk[x, y + 1, z] == 0.0:
                    vertex_x += translate(cube_vertices[FACE_TOP], x, y, z)
                    vertex_n += cube_normals[FACE_TOP]
                if y > 0 and chunk[x, y - 1, z] == 0.0:
                    vertex_x += translate(cube_vertices[FACE_BOTTOM], x, y, z)
                    vertex_n += cube_normals[FACE_BOTTOM]
                if z < 15 and chunk[x, y, z + 1] == 0.0:
                    vertex_x += translate(cube_vertices[FACE_BACK], x, y, z)
                    vertex_n += cube_normals[FACE_BACK]
                if z > 0 and chunk[x, y, z - 1] == 0.0:
                    vertex_x += translate(cube_vertices[FACE_FRONT], x, y, z)
                    vertex_n += cube_normals[FACE_FRONT]
    return (np.array(vertex_x, dtype=np.float32), np.array(vertex_n, dtype=np.float32))

chunk = generate_chunk(420)
mesh_x, mesh_n = compute_mesh(chunk)

# Each face has 18 coordinates (6 * 3). Print the number
# of visible faces in the computed mesh.
print(len(mesh_x) // 18, "faces")
