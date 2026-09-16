import time

from geometry_msgs.msg import Twist

from models import ControlCommand


class CmdVelBlock:
    """Only responsible for publishing robot velocity commands."""

    def __init__(self, node, leader_topic: str, follower_topic: str) -> None:
        self.leader_pub = node.create_publisher(Twist, leader_topic, 10)
        self.follower_pub = node.create_publisher(Twist, follower_topic, 10)

    def publish(self, command: ControlCommand) -> None:
        leader_cmd = Twist()
        leader_cmd.linear.x = float(command.leader_v)
        leader_cmd.angular.z = float(command.leader_u)
        self.leader_pub.publish(leader_cmd)

        follower_cmd = Twist()
        follower_cmd.linear.x = float(command.follower_v)
        follower_cmd.angular.z = float(command.follower_u)
        self.follower_pub.publish(follower_cmd)

    def stop(self) -> None:
        stop = Twist()
        self.leader_pub.publish(stop)
        self.follower_pub.publish(stop)

    def shutdown(self, repeats: int = 5, delay: float = 0.1) -> None:
        for _ in range(repeats):
            self.stop()
            time.sleep(delay)
