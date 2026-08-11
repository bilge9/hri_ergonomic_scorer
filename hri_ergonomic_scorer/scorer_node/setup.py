import os
from setuptools import find_packages, setup

package_name = 'hri_ergonomic_scorer'

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), ['launch/reba_dashboard.launch.py']),
]

# The DEBA checkpoint is a training artefact, not source: it is produced by
# generate_deba_dataset.py + train_deba_model.py and must be regenerated
# whenever the REBA teacher changes. Install it only if it is present, so a
# clean checkout still builds and runs REBA-only.
_model = os.path.join(package_name, 'deba_model.pth')
if os.path.exists(_model):
    data_files.append((os.path.join('share', package_name, 'models'), [_model]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bilge',
    maintainer_email='bilgenurdemirel49@gmail.com',
    description=('REBA- and DEBA-inspired ergonomic risk assessment from '
                 'hri_msgs 3D skeleton tracking, for Vulcanexus HRI.'),
    license='Apache-2.0',
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
            'annotation_recorder = hri_ergonomic_scorer.annotation_recorder:main',
        ],
    },
)
