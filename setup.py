import setuptools

ver_globals = {}
with open("radere/version.py") as fp:
    exec(fp.read(), ver_globals)
version = ver_globals["version"]

setuptools.setup(
    name="radere",
    version=version,
    author="Brett Viren",
    author_email="brett.viren@gmail.com",
    description="Calculate response functions with FDM field calculations",
    url="https://brettviren.github.io/radere",
    packages=setuptools.find_packages(),
    python_requires='>=3.5',
    install_requires=[
        "click",         # CLI
        # "h5py",          # support HDF5 files or
        "numpy",         # .npz, need numpy in general
        "scipy",         # better fft on cpu than numpy
        "matplotlib",
        "torch",         # for CPU/GPU
        # "desolver",      # candidate for rk, needs post-inst hack/fix 
        # "torchdiffeq",   # candidate for rk
        # "pyevtk",        # for optional export to VTK
        "pytest",
    ],
    entry_points = dict(
        console_scripts = [
            'radere = radere.__main__:main',
        ]
    ),
    #include_package_data=True,
)

