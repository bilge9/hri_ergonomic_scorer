from setuptools import find_packages, setup

package_name = 'hri_ergonomic_scorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bilge',
    maintainer_email='bilgenurdemirel49@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scorer_node = hri_ergonomic_scorer.ergonomic_scorer_node:main',
            'fiware_reba_bridge = hri_ergonomic_scorer.fiware_reba_bridge:main',
            'skeleton_overlay = hri_ergonomic_scorer.skeleton_overlay_node:main',
        ],
    },
)
