## CREATE PATHS / DIRECTORIES 
class Mkdirs:
    def __init__(self,path_home = False):
        # path to home directory 
        if not path_home: 
            self.path = os.getcwd()
        else:
            self.path = path_home
        # set directory
        os.chdir(self.path)
        #Check if data directories exist - creat directories if needed
        self.paths = {}
        self.paths['data'] = './dataXX'
        self.paths['census'] = self.paths['data'] + '/census'
        self.paths['onet'] = self.paths['data'] + '/onet'

        [os.makedirs(pth, exist_ok=True) for pth in paths.values()]     
  


