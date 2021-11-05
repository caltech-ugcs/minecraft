import numpy as np
from OpenGL import *
from OpenGL.GL import *
from noise import pnoise2
from .Cube import *
import glm

class Chunk:
    def __init__(self, world, u: int, v: int):
        self.location = (u, v)
        self.world = world
        self.blocks = np.zeros((16, 256, 16), dtype=np.int16)
        self.modified = True

        self.vao = glGenVertexArrays(1)
        self.vbo_x = glGenBuffers(1)
        self.vbo_n = glGenBuffers(1)
        self.n_vertices = 0
        self.model_x = glm.translate(glm.vec3(u * 16 - 8, 0, v * 16 - 8))
        self.generate()
        
    def generate(self):
        """
        Generate a single chunk with the given seed according to the following algorithm:
        For each (x, z) coordinate, compute the perlin noise function. Fill in all blocks
        with a height less than ground + amplitude * noise(x, z).
        """
        (u, v) = self.location
        amplitude = 10
        ground = 128
        scale = 3
        for x in range(0, 16):
            for z in range(0, 16):
                noise = pnoise2((x / 16 + u) / scale, (z / 16 + v) / scale, octaves=2, base=self.world.seed)
                height = int(ground + amplitude * noise)
                for y in range(0, 256):
                    if y < height:
                        self.blocks[x,y,z] = 1

    def block(self, x: int, y: int, z: int):
        (u, v) = self.location
        x = 16 * u + x
        z = 16 * v + z
        if x // 16 == u and z // 16 == v:
            return self.blocks[x % 16, y, z % 16]
        else:
            if (x // 16, z // 16) in self.world.chunks.keys():
                chunk = self.world.chunks[(x // 16, z // 16)]
                return chunk.blocks[x % 16, y, z % 16]
            else:
                return -1

    def compute_mesh(self):
        """
        Compute the vertex and normal buffers for the mesh containing all currently
        visible faces (all faces adjacent to air) 
        """
        vertex_x = []
        vertex_n = []
        chunk = self.blocks
        for x in range(0, 16):
            for z in range(0, 16):
                for y in range(0, 256):
                    if chunk[x,y,z] == 0.0:
                        continue
                    if self.block(x + 1, y, z) == 0:
                        vertex_x += translate(cube_vertices[FACE_RIGHT], x, y, z)
                        vertex_n += cube_normals[FACE_RIGHT]
                    if self.block(x - 1, y, z) == 0:
                        vertex_x += translate(cube_vertices[FACE_LEFT], x, y, z)
                        vertex_n += cube_normals[FACE_LEFT]
                    if y < 255 and chunk[x, y + 1, z] == 0:
                        vertex_x += translate(cube_vertices[FACE_TOP], x, y, z)
                        vertex_n += cube_normals[FACE_TOP]
                    if y > 0 and chunk[x, y - 1, z] == 0:
                        vertex_x += translate(cube_vertices[FACE_BOTTOM], x, y, z)
                        vertex_n += cube_normals[FACE_BOTTOM]
                    if self.block(x, y, z + 1) == 0:
                        vertex_x += translate(cube_vertices[FACE_BACK], x, y, z)
                        vertex_n += cube_normals[FACE_BACK]
                    if self.block(x, y, z - 1) == 0:
                        vertex_x += translate(cube_vertices[FACE_FRONT], x, y, z)
                        vertex_n += cube_normals[FACE_FRONT]
        vertex_x = np.array(vertex_x, dtype=np.float32)
        vertex_n = np.array(vertex_n, dtype=np.float32)
        self.n_vertices = len(vertex_x) // 3
        
        glBindVertexArray(self.vao)
       
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_x)
        glBufferData(GL_ARRAY_BUFFER, vertex_x.nbytes, vertex_x, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_n)
        glBufferData(GL_ARRAY_BUFFER, vertex_n.nbytes, vertex_n, GL_STATIC_DRAW)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
    def update(self):
        """
        Perform physics stuff.
        """
        if self.modified:
            self.compute_mesh()
            self.modified = False