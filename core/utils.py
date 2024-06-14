import zipfile

def extract(source_path, destination_path):
    
    with zipfile.ZipFile(source_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

class standardize:
    def __init__(self, mean_val=None,std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):        
        return (x - self.mean_val)/ self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val

