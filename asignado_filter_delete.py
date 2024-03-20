class disp_fig(object):
    def __init__(self, arr):
        self.array = arr
        
        # Reshape numpy video array
        self.video = arr.reshape(8,152,152)