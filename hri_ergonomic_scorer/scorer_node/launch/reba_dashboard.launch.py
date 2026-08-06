import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Ana REBA/DEBA Hesaplama Düğümü
        Node(
            package='hri_ergonomic_scorer',
            executable='scorer_node', # Senin setup.py'daki ismin
            name='ergonomic_scorer_node',
            output='screen',
            parameters=[
                {'verbose_logging': False}
            ]
        ),
        
        # 2. FIWARE Veritabanı Köprüsü
        Node(
            package='hri_ergonomic_scorer',
            executable='fiware_reba_bridge', # Senin setup.py'daki ismin
            name='fiware_reba_bridge_ld',
            output='screen'
        ),
        
        # 3. Grafana için İskelet Çizim (Overlay) Düğümü
        Node(
            package='hri_ergonomic_scorer',
            executable='skeleton_overlay', # Senin setup.py'daki ismin
            name='skeleton_overlay_node',
            output='screen'
        ),

        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server',
            output='screen',
            parameters=[
                {'port': 8080}
            ]
        )
    ])