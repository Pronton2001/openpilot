import math
from multiprocessing import Queue

# from metadrive.envs.scenario_env import ScenarioEnv
# from metadrive.component.sensors.base_camera import _cuda_enable

from openpilot.tools.sim.bridge.common import SimulatorBridge
# from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.scenarionet_world import ScenarioNetWorld
from openpilot.tools.sim.lib.camerad import W, H


class ScenarioNetLidarBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=math.inf, test_run=False):
    super().__init__(dual_camera, high_quality)

    self.should_render = False
    self.test_run = test_run
    self.test_duration = test_duration if self.test_run else math.inf

  def spawn_world(self, queue: Queue):
    dataset = '/media/ml4u/Challenge-4TB/scenarionet/filtered_dataset/thresh_0_2/all/av2_train/'
    config = dict(
      manual_control = False,
      # reactive_traffic = False,
      data_directory= dataset,
      num_scenarios=10000,
      even_sample_vehicle_class = False,

      use_render=self.should_render, # always False
      vehicle_config=dict(side_detector=dict(num_lasers=0)),
      # sensors=sensors,
      # image_on_cuda=_cuda_enable,
      # image_on_cuda=True,
      # image_observation=True, # False
      interface_panel=[],
      out_of_route_done=False,
      # on_continuous_line_done=False,
      crash_vehicle_done=False,
      crash_object_done=False,
      arrive_dest_done=False,
      # traffic_density=0.1, # traffic is incredibly expensive
      # map_config=create_map(),
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False,
      show_logo=False,
      # anisotropic_filtering=False # local metadrive(pulled 18/11/2024) does not work with this
    )

    return ScenarioNetWorld(queue, config, self.test_duration, self.test_run, self.dual_camera)
