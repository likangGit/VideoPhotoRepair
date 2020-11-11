import os
from abc import ABC, abstractmethod
class OP(ABC):
    def __init__(self, outputdir):
        self.outputdir=outputdir
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
    
    @abstractmethod
    def exec(self, param_dict):
        raise NotImplementedError