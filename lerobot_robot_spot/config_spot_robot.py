# config_spot_robot.py
from dataclasses import dataclass, field
from typing import Dict, List

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("spot_robot")
@dataclass
class SpotRobotConfig(RobotConfig):
    """
    Configuration for the Boston Dynamics Spot robot in LeRobot.

    Fields:
        hostname: IP or hostname of Spot.
        username/password: Credentials for Spot's user account.
        update_rate_hz: Target control loop frequency for your control loop.
        use_front_cameras: Whether to stream front stereo pair.
        use_arm_camera: Whether to stream the arm/hand camera.
        image_sources: Optional explicit list of Spot image sources to use.
                       If empty, will be derived from the *_cameras flags.
        image_format: Spot image format (e.g. FORMAT_RGB_U8).
        image_width, image_height: Optional resize; 0 means native size.
    """

    hostname: str = "192.168.80.3"
    username: str = "admin"
    password: str = "password"

    update_rate_hz: float = 10.0

    # High-level camera toggles
    use_front_cameras: bool = True
    use_arm_camera: bool = True

    # Explicit Spot image source names (overrides flags if not empty)
    image_sources: List[str] = field(default_factory=list)

    # Image format and size hints. These map to Spot's ImageRequest fields.
    image_format: int = 0  # Will be interpreted as FORMAT_RGB_U8 in SpotRobot
    image_width: int = 0   # 0 => keep native width
    image_height: int = 0  # 0 => keep native height

    # Force-take the lease from another client instead of a normal acquire.
    force_take_lease: bool = False

    # You can add more config fields later (e.g. max velocities, arm limits).
    extra: Dict[str, float] = field(default_factory=dict)


