import numpy as np

class Chunk:
    def __init__(self, world):
        self.world = world
        self.blocks = np.zeros((16, 256, 16), dtype=np.int16)
        self.modified = True
        self.vertex_buffer = None
        self.normal_buffer = None

    def generate_blocks(self):
        """
        Fills self.blocks using TBD generation algorithm
        """
        pass

    def regenerate_mesh(self):
        if self.modified == False:
            return

    def update(self):
        """
        Perform physics stuff.
        """
        pass

    def draw(self):
        pass