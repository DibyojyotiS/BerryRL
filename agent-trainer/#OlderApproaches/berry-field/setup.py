from setuptools import setup

setup(
    name='berry_field',
    version='0.0.1',

    # removed pickel, time from this list since this cannot be installed by pip
    install_requires=['gym', 'numpy', 'pyglet']
)
