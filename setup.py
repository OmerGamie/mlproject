from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    
    Reads requirements.txt and returns a list of requirements
    
    '''
    requirements=[]
    with open(file_path,'r') as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements
        
             
setup(
    name='mlproject',
    version='0.1.0',
    author='Omer',
    author_email='3mer86@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'), # list of dependencies
    
    
)