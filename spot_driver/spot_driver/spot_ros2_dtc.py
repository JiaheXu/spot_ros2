#!/usr/bin/env python3
# Debug
# from ros_helpers import *
import logging
import os
import tempfile
import threading
import time
import traceback
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import bdai_ros2_wrappers.process as ros_process
import builtin_interfaces.msg
import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
from bdai_ros2_wrappers.node import Node
from bdai_ros2_wrappers.single_goal_action_server import (
    SingleGoalActionServer,
)
from bdai_ros2_wrappers.single_goal_multiple_action_servers import (
    SingleGoalMultipleActionServers,
)
from bosdyn.api import (
    geometry_pb2,
    gripper_camera_param_pb2,
    manipulation_api_pb2,
    robot_command_pb2,
    trajectory_pb2,
    world_object_pb2,
)
from bosdyn.api.geometry_pb2 import Quaternion, SE2VelocityLimit
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.api.spot.choreography_sequence_pb2 import Animation, ChoreographySequence, ChoreographyStatusResponse
from bosdyn.client import math_helpers
from bosdyn.client.exceptions import InternalServerError
from bosdyn_api_msgs.math_helpers import bosdyn_localization_to_pose_msg
from bosdyn_msgs.conversions import convert
from bosdyn_msgs.msg import (
    ArmCommandFeedback,
    Camera,
    FullBodyCommand,
    FullBodyCommandFeedback,
    GripperCommandFeedback,
    Logpoint,
    ManipulationApiFeedbackResponse,
    MobilityCommandFeedback,
    PtzDescription,
    RobotCommand,
    RobotCommandFeedback,
    RobotCommandFeedbackStatusStatus,
)

from google.protobuf import duration_pb2

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import (arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2,
                        trajectory_pb2)
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient



from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Twist,
)
from rclpy import Parameter
from rclpy.action import ActionServer
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import CallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock
from rclpy.impl import rcutils_logger
from rclpy.publisher import Publisher
from rclpy.timer import Rate
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, Trigger

import spot_driver.robot_command_util as robot_command_util

# DEBUG/RELEASE: RELATIVE PATH NOT WORKING IN DEBUG
# Release
from spot_driver.ros_helpers import (
    get_from_env_and_fall_back_to_param,
)
from spot_msgs.action import (  # type: ignore
    ExecuteDance,
    Manipulation,
    NavigateTo,
    Trajectory,
)
from spot_msgs.action import (  # type: ignore
    RobotCommand as RobotCommandAction,
)
from spot_msgs.msg import (  # type: ignore
    Feedback,
    LeaseArray,
    LeaseResource,
    Metrics,
    MobilityParams,
)
from spot_msgs.srv import (  # type: ignore
    ChoreographyRecordedStateToAnimation,
    ChoreographyStartRecordingState,
    ChoreographyStopRecordingState,
    ClearBehaviorFault,
    DeleteLogpoint,
    DeleteSound,
    Dock,
    GetChoreographyStatus,
    GetGripperCameraParameters,
    GetLEDBrightness,
    GetLogpointStatus,
    GetPtzPosition,
    GetVolume,
    GraphNavClearGraph,
    GraphNavGetLocalizationPose,
    GraphNavSetLocalization,
    GraphNavUploadGraph,
    InitializeLens,
    ListAllDances,
    ListAllMoves,
    ListCameras,
    ListGraph,
    ListLogpoints,
    ListPtz,
    ListSounds,
    ListWorldObjects,
    LoadSound,
    OverrideGraspOrCarry,
    PlaySound,
    RetrieveLogpoint,
    SetGripperAngle,
    SetGripperCameraParameters,
    SetLEDBrightness,
    SetLocomotion,
    SetPtzPosition,
    SetVelocity,
    SetVolume,
    StoreLogpoint,
    TagLogpoint,
    UploadAnimation,
    UploadSequence,
)
from spot_msgs.srv import (  # type: ignore
    RobotCommand as RobotCommandService,
)
from spot_wrapper.cam_wrapper import SpotCamCamera, SpotCamWrapper
from spot_wrapper.wrapper import SpotWrapper

MAX_DURATION = 1e6
COLOR_END = "\33[0m"
COLOR_GREEN = "\33[32m"
COLOR_YELLOW = "\33[33m"

from spot_driver.spot_ros2 import (
    set_node_parameter_from_parameter_list,
    WaitForGoal,
    GoalResponse,
    Response,
    Request,
    SpotROS
)


from geometry_msgs.msg import PointStamped, TwistStamped, Twist
from std_msgs.msg import UInt8, Bool, String
from sensor_msgs.msg import Joy
# from airlab_msgs.msg import AIRLABModes
from std_msgs.msg import String, Float32, Int8, UInt8, Bool, UInt32MultiArray, Int32
from std_srvs.srv import SetBool
from rclpy.client import Client
from std_srvs.srv import Trigger
import rclpy
import time



from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d
import transforms3d as t3d
import math

# for GPIO
#import RPi.GPIO as GPIO

class SpotROS_DTC(SpotROS):

    def __init__(self, **kwargs):
        super().__init__( **kwargs )
        self.get_logger().info(COLOR_GREEN + "Hi from spot_driver_dtc." + COLOR_END)
        _, __ = self.spot_wrapper.spot_arm.gripper_open()
        # self.command_client = self.spot_wrapper.ensure_client(RobotCommandClient.default_service_name)
        self.arm_status = False
        self.last_manual_mode = self.get_clock().now()
        self.manual_override_engaged = False
        
        self.claim_status = False 
        self.power_on_status = False
        self.stand_status = False

        
        # Pin Definitions
        self.estop_on = False
        self.estop_prev_value = False
        self.estop_input_pin = 22  # BOARD pin 22
        
        self.on_jetson = False
        if( self.on_jetson ):

            GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
            GPIO.setup(self.estop_input_pin, GPIO.IN)  # set pin as an input pin

        # output
        self.tier2_estop_pub = self.create_publisher(Bool, "/tier2_estop", 1)

        # self.armed_status_pub = self.create_publisher(Bool, "status/arm", 1)
        # self.spot_control_pub = self.create_publisher(Twist, "cmd_vel", 1)
        self.count = 30
        self.arm_control()
        self.timer = self.create_timer(0.1, self.status_timer_callback)

        # self.timer2 = self.create_timer(1, self.arm_control)
        # self.spot_wrapper is robot in examples
    def echo(self, msg_str):
        self.get_logger().info(msg_str)


    def status_timer_callback(self):

        
        if( self.on_jetson ):
            value = GPIO.input(self.estop_input_pin)
            if value == GPIO.HIGH:
                self.estop_on = True
            else:
                self.estop_on = False

        tier2_estop_msg = Bool()
        tier2_estop_msg.data = self.estop_on
        self.tier2_estop_pub.publish(tier2_estop_msg)

        self.count -= 1
        if(self.count == 0):
            self.look_at( np.array([1., -1., 0. ] ))
        # arm_msg = Bool()
        # arm_msg.data = self.arm_status
        # self.armed_status_pub.publish(arm_msg)

        return

    def arm_control(self):
        # self.spot_wrapper.time_sync.wait_for_sync()

        assert self.spot_wrapper.has_arm(), 'Robot requires an arm to run this example.'
        
        self.get_logger().info(COLOR_GREEN + "has_arm." + COLOR_END)
        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not self.spot_wrapper.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                        'such as the estop SDK example, to configure E-Stop.'
        
        self.get_logger().info(COLOR_GREEN + "not estopped." + COLOR_END)
        _, __ = self.spot_wrapper.spot_arm.gripper_open()

        # unstow = RobotCommandBuilder.arm_ready_command()
        # unstow_command_id = self.command_client.robot_command(unstow)
        self.spot_wrapper.spot_arm.arm_unstow()

    def get_lookat_position(self, position, target):
        # we only rotate z and y axis

        forward = target - position
        forward = forward / np.linalg.norm(forward)
        x, y, z = forward[0], forward[1], forward[2]
        
        yaw = math.atan2(y, x)
        
        pitch = -math.atan2(z, math.sqrt(x*x + y*y) )

        print("pitch: ", pitch)
        Rx = t3d.euler.euler2mat(0, 0, 0)
        Ry = t3d.euler.euler2mat(0, pitch, 0)
        Rz = t3d.euler.euler2mat(0, 0, yaw)
        R = Rz @ Ry
        # R = Ry 
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()
        trans = np.eye(4)
        trans[:3,:3] = R
        trans[0:3,3] = position
        return trans
    
    def look_at(self, position):


        target = np.array( [0.5, 0, 0.5] )
        goal = self.get_lookat_position( position, target)
        # move to goal 

# need to use customized frame in the future
# https://github.com/bdaiinstitute/spot_ros2/commit/6410c75e43a439cd22e3f51b60f7710c8338dfc2


@ros_process.main(prebaked=False)
def main(args: Optional[List[str]] = None) -> None:
    ros_process.spin(SpotROS_DTC)


if __name__ == "__main__":
    main()
