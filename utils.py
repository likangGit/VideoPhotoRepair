import os
from abc import ABC, abstractmethod
class OP(ABC):
    def __init__(self, outputdir):
        self.inputdir=''
        self.outputdir=outputdir
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
    
    @abstractmethod
    def exec(self):
        raise NotImplementedError