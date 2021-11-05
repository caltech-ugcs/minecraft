from .Chunk import Chunk

class World:
    """
    A class to represent a single 'World' in the Metaverse.
    """

    # The seed that determines random world generation.
    seed: int

    # The chunks that are currently generated for this world.
    chunks: dict[tuple[int, int], Chunk]

    def __init__(self, seed):
        self.seed = seed
        self.chunks = {}
        
    def generate(self, u: int, v: int):
        """
        Generate a new Chunk with the given chunk coordinates.

        Parameters:
            u (int): the chunk coordinate in the x direction.
            v (int): the chunk coordinate in the z direction.
        """
        if not (u, v) in self.chunks.keys():
            self.chunks[(u, v)] = Chunk(self, u, v)

    def update(self):
        """
        Update all generated chunks
        """
        for _, chunk in self.chunks.items():
            chunk.update()

    def draw(self):
        """
        Draw all generated chunks using OpenGL
        """
        for _, chunk in self.chunks.items():
            chunk.draw()