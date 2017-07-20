from setuptools import setup

setup(name='fami',
      version='0.1',
      description='Federated Agents Making Inferences about games',
      url='http://github.com/JoeOsborn/mechlearn',
      author='Joseph C. Osborn and Adam Summerville',
      author_email='jcosborn@ucsc.edu',
      license='MIT',
      packages=['nes', 'util'],
      install_requires=[
          'cv2wrap',
          'datrie',
          'numpy',
          'scipy',
          'statsmodels',
          'pyzmq',
          'pymc3',
          'Pillow',
          'networkx',
          'matplotlib'
      ],
      # TODO: add fceulib here
      # dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0']
      zip_safe=False)
