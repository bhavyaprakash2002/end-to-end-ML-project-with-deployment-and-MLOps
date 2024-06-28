from setuptools import find_packages, setup
from typing import List


hyphen_e_dot = "-e ."
def get_requirements(file_path:str)->List[str]:
    # it will return list of requirements
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[i.replace("\n"," ") for i in requirements]
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
            


setup(
name = "end to end ML project with deployment and MLOps",
version="0.0.1",
author="Bhavya Prakash",
author_email="bhavyaprakash2002@gmail.com",
packages=find_packages(),
# install_requires = ['pandas', 'numpy', 'matplotlib', 'seaborn']
install_requires = get_requirements('requirements.txt')
)