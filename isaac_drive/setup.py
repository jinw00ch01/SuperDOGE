from glob import glob

from setuptools import setup

package_name = "isaac_drive"

setup(
    name=package_name,
    version="0.0.1",
    packages=[
        package_name,
        f"{package_name}.navigation",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # 학습한 구동 조절기 가중치 (isaac_sim/rl/train_drive.py 의 결과).
        (f"share/{package_name}/policies", glob("policies/*.npz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jimjimjinger",
    maintainer_email="jimjimjinger@users.noreply.github.com",
    description="Drive package: autonomous/manual driving flow + rover drive execution + navigation primitives.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "coverage_node = isaac_drive.coverage_node:main",
            "odom_to_estimated_pose = isaac_drive.odom_to_estimated_pose:main",
            "hazard_map_node = isaac_drive.hazard_map_node:main",
            "drive_governor_node = isaac_drive.drive_governor_node:main",
            "self_right_node = isaac_drive.self_right_node:main",
        ],
    },
)
