
class World:
    def __init__(self, seed):
        self.seed = seed
        self.chunks = {}
        self.visible_chunks = {}
        
    def get_chunk(self, u, v):
        return self.chunks[(u, v)]
    
    def generate_chunk(self, u, v):
        """
        Create a new chunk and add it to the map
        """
        pass


    def update():
        """
        Run physics engine.
        Update self.visible_chunks
        """
        pass

    def draw():
        """
        Render self.visible_chunks using OpenGL
        """
        # visible_chunks =
        pass