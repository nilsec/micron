import numpy as np
import operator
import logging

logger = logging.getLogger(__name__)

def dda_round(x):
    return (x + 0.5).astype(int) 

class DDA3:
    def __init__(self, start, end, scaling=np.array([1,1,1])):
        assert(start.dtype == int)
        assert(end.dtype == int)

        self.scaling = np.array(scaling)

        self.start = np.array((start * scaling), dtype=float)
        self.end = np.array((end * scaling), dtype=float)
        self.line = [np.array(dda_round(self.start/self.scaling))]
        
        self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)), 
                                                  key=operator.itemgetter(1))

        try:
            self.dv = (self.end - self.start) / self.max_length
        except RuntimeWarning:
            logger.debug("max length: " + str(self.max_length))
            raise ValueError


    def draw(self):
        # We interpolate in physical space to find the shortest distance
        # linear interpolation but the path is represented in voxels
        for step in range(int(self.max_length)):
            step_point_rescaled = np.array(dda_round(dda_round((step + 1) * self.dv + self.start)/self.scaling)) 
            if not np.all(step_point_rescaled == self.line[-1]):
                self.line.append(step_point_rescaled)

        assert(np.all(self.line[-1] == dda_round(self.end/self.scaling)))
        return self.line
