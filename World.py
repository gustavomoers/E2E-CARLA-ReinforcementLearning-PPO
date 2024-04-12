import pygame
import carla
from Utils.synch_mode import CarlaSyncMode
import Controller.PIDController as PIDController
import time
from Utils.utils import *
import math
import gym
import gymnasium as gym
from gymnasium import spaces
# from Utils.HUD import HUD as HUD
from Utils.CubicSpline.cubic_spline_planner import *
import csv
from Utils.HUD_visuals import *



class World(gym.Env):
    def __init__(self, client, carla_world, hud, args, visuals=False):
        self.world = carla_world
        self.client = client
        self.map = self.world.get_map()
        self.hud = hud
        self.args = args
        self._gamma = args.gamma
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        self.control_mode = args.control_mode
        self.controller = None
        self.control_count = 0.0
        self.random_spawn = 0
        self.world.on_tick(hud.on_world_tick)
        self.im_width = 640
        self.im_height = 480
        self.episode_start = 0
        self.visuals = visuals
        self.episode_reward = 0
        self.player = None
        self.parked_vehicle = None
        self.collision_sensor = None
        self.camera_rgb = None
        self.lane_invasion = None
        self._autopilot_enabled = False
        self._control = carla.VehicleControl()
        self.max_dist = 4.5
        self.counter = 0
        self.frame = None
        self.delta_seconds = 1.0 / args.FPS
        self.last_v = 0
        self.last_y = 0
        self.distance_parked = 100
        self.ttc_trigger = 1.0
        self.episode_counter = 0
        self.steer = 0
        self.save_list = []
        # self.file_name = 'F:/E2E-CARLA-ReinforcementLearning-PPO/logs/1709073714-working-50kmh/evaluation/logger.csv'
        self.logger = False

        #VISUAL PYGAME
        self._weather_presets = find_weather_presets()
        self.visuals = visuals
        self.collision_sensor_hud = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None


        ## RL STABLE BASELINES
        self.action_space = spaces.Box(low=-1, high=1,shape=(2,),dtype="float")
        self.observation_space = spaces.Box(low=-0, high=255, shape=(128, 128, 1), dtype=np.uint8)

        # self.visuals = visuals
        # if self.visuals:
        #     self._initiate_visuals()

        self.global_t = 0 # global timestep


    def append_to_csv(self,file_name, data):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)


    def reset(self, seed=None):

        self.destroy()

        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1/self.args.FPS))
        self.episode_reward = 0
        self.desired_speed = self.args.desired_speed

        if self.visuals:
            # Keep same camera config if the camera manager exists.
            cam_index = self.camera_manager.index if self.camera_manager is not None else 0
            cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
            # # # Get a random blueprint.
        

        self.episode_counter += 1

        if self.logger:
            self.append_to_csv(file_name=self.file_name, data=self.save_list)
        self.save_list = []


        # CREATING ACTORS

        self.create_actors()


        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_roration = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y
        current_yaw = wrap_angle(current_roration.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        frame, current_timestamp = self.hud.get_simulation_information()
        self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
        self.episode_start = time.time()


        if self.visuals:
            self.collision_sensor_hud = CollisionSensor(self.player, self.hud)
            self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
            self.gnss_sensor = GnssSensor(self.player)
            self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
            self.camera_manager.transform_index = cam_pos_index
            self.camera_manager.set_sensor(cam_index, notify=False)



        self.world.tick()             
        self.clock = pygame.time.Clock()

            
        ttc = self.time_to_collison()

        while ttc > self.ttc_trigger: #player_position < parked_position - self.realease_position:
            

            self.clock.tick_busy_loop(self.args.FPS)

            if self.visuals:  
                self.display = pygame.display.set_mode(
                            (self.args.width, self.args.height),
                            pygame.HWSURFACE | pygame.DOUBLEBUF)
                self.display.fill((0,0,0))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False

                self.tick(self.clock)
                self.render(self.display)
                pygame.display.update()
           

            if self.parse_events(clock=self.clock, action=None):
                 return
            
            velocity_vec_st = self.player.get_velocity()
            current_speed = math.sqrt(velocity_vec_st.x**2 + velocity_vec_st.y**2 + velocity_vec_st.z**2)

            ttc = self.time_to_collison()
   
            snapshot, image_rgb, lane, collision = self.synch_mode.tick(timeout=10.0)

            self.get_observation()


            if image_rgb is not None:
                img = process_img2(self, image_rgb)
           
  

        last_transform = self.player.get_transform()
        last_location = last_transform.location
        self.last_y = last_location.y
        self.last_v = current_speed
        print(current_speed)

        return img, {}


    def tick(self, clock):
        self.hud.tick(self, clock)


    def destroy(self):
        self.world.tick()
            
        actors = [
            self.player,
            self.collision_sensor,
            self.camera_rgb,
            self.lane_invasion,
            self.parked_vehicle]    

        if self.collision_sensor_hud is not None:
            actors.append(self.collision_sensor_hud.sensor)
            actors.append(self.lane_invasion_sensor.sensor)
            actors.append(self.gnss_sensor.sensor)
            actors.append(self.camera_manager.sensor)           
                           
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                    self.world.tick()
                except:
                    pass



    def step(self, action):


        self.reward = 0
        done = False
        cos_yaw_diff = 0
        dist = 0
        collision = 0
        lane = 0
        traveled = 0

        if action is not None:

            self.counter += 1
            self.global_t += 1

            
            self.clock.tick_busy_loop(self.args.FPS)
            
            if self.apply_vehicle_control(action):
                return
            
            
            
            snapshot, image_rgb, lane, collision = self.synch_mode.tick(timeout=10.0)
            
            
            self.get_observation()


            cos_yaw_diff, dist, collision, lane, traveled = self.get_reward_comp(self.player, self.spawn_waypoint, collision, lane)
            
            
            self.reward = self.reward_value(cos_yaw_diff, dist, collision, lane, traveled)


            if self.visuals:
                self.display.fill((0,0,0))
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False                

                self.tick(self.clock)
                self.render(self.display)
                pygame.display.flip()

     
            self.episode_reward += self.reward

            if image_rgb is not None:
                image = process_img2(self, image_rgb)
                    
            
            if dist > self.max_dist:
                done=True


            vehicle_location = self.player.get_location()
            y_vh = vehicle_location.y
            if y_vh > float(self.args.spawn_y)+self.distance_parked+15:
                self.reward += 50
                print("episode ended by reaching goal position")
                self.hud.notification("episode ended by reaching goal position")
                done=True


            truncated = False
 

            if collision == 1:
                done=True
                print("Episode ended by collision")
                self.hud.notification("episode ended by collision")
            
            if lane == 1:
                done = True
                self.reward -= 50
                print("Episode ended by lane invasion")
                self.hud.notification("episode ended by lane invasion")
    
            if dist > self.max_dist:
                done=True
                self.reward -= 50
                print(f"Episode  ended with dist from waypoint: {dist}")
                self.hud.notification(f"Episode  ended with dist from waypoint: {dist}")

            velocity_vec_st = self.player.get_velocity()
            current_speed = math.sqrt(velocity_vec_st.x**2 + velocity_vec_st.y**2 + velocity_vec_st.z**2)
            if current_speed < 0.1:
                self.hud.notification("episode ended by fully stopping")
                done=True
                

        return image, self.reward, done, truncated, {}
    



    def get_reward_comp(self, vehicle, waypoint, collision, lane):
        vehicle_location = vehicle.get_location()
        x_wp = waypoint.transform.location.x
        y_wp = waypoint.transform.location.y

        x_vh = vehicle_location.x
        y_vh = vehicle_location.y

        wp_array = np.array([x_wp])
        vh_array = np.array([x_vh])

        dist = abs(np.linalg.norm(wp_array - vh_array))

        vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
        wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
        cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

        collision = 0 if collision is None else 1

        if lane is not None:
            lane_types = set(x.type for x in lane.crossed_lane_markings)
            text = ['%r' % str(x).split()[-1] for x in lane_types]
            lane = 1 if text[0] == "'Solid'" else 0
        
        elif lane is None:
            lane=0

        # lane = 0 if lane is None else 1

        traveled = y_vh - self.last_y
        # print(traveled)
 


        # finish = 1 if y_vh > -40 else 0
        
        return cos_yaw_diff, dist, collision, lane, traveled
    
    def reward_value(self, cos_yaw_diff, dist, collision, lane, traveled, lambda_1=1, lambda_2=1, lambda_3=100, lambda_4=5, lambda_5=0.5):
    
        reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision) - (lambda_4 * lane) + (lambda_5 * traveled)
        
        return reward
    


    def time_to_collison(self):

         # EGO information
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_x = current_location.x
        current_y = current_location.y
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
       

        #Parked vehicle information
        parked_transform = self.parked_vehicle.get_transform()
        velocity_parked = self.parked_vehicle.get_velocity()
        parked_location = parked_transform.location
        parked_x = parked_location.x
        parked_y = parked_location.y
        parked_speed = math.sqrt(velocity_parked.x**2 + velocity_parked.y**2 + velocity_parked.z**2)

        dist = np.sqrt((parked_y-current_y)**2 + (current_x-parked_x)**2)
        rel_speed = current_speed - parked_speed

        ttc = dist/rel_speed

        return np.abs(ttc)


    def parse_events(self, action, clock):

        if not self._autopilot_enabled:
            # Control loop
            # get waypoints
            current_location = self.player.get_location()
            velocity_vec = self.player.get_velocity()
            current_transform = self.player.get_transform()
            current_location = current_transform.location
            current_rotation = current_transform.rotation
            current_x = current_location.x
            current_y = current_location.y
            current_yaw = wrap_angle(current_rotation.yaw)
            current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
            # print(f"Control input : speed : {current_speed}, current position : {current_x}, {current_y}, yaw : {current_yaw}")
            frame, current_timestamp =self.hud.get_simulation_information()
            ready_to_go = self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
            
            if ready_to_go:
                if self.control_mode == "PID":
                    current_location = self.player.get_location()
                    current_waypoint = self.map.get_waypoint(current_location).next(self.waypoint_resolution)[0]
                    # print(current_waypoint.transform.location.x-current_x)
                    # print(current_waypoint.transform.location.y-current_y)            
                    waypoints = []
                    for i in range(int(self.waypoint_lookahead_distance / self.waypoint_resolution)):
                        waypoints.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y, self.desired_speed])
                        current_waypoint = current_waypoint.next(self.waypoint_resolution)[0]


                # print(f'wp real: {waypoints}')
                if action is not None:
                    waypoints_RL = self.get_cubic_spline_path(action, current_x=current_x, current_y=current_y)
                    self.print_waypoints(waypoints_RL)
                    # print(waypoints_RL)
                    self.controller.update_waypoints(waypoints_RL)
                else:
                    self.print_waypoints(waypoints)
                    self.controller.update_waypoints(waypoints)  

                self.controller.update_controls()
                self._control.throttle, self._control.steer, self._control.brake = self.controller.get_commands()
                # print(self._control)
                self.player.apply_control(self._control)
                self.control_count += 1

    
    def apply_vehicle_control(self, action):

        self.steer = action[0]
        print(f'steer = {self.steer}')
        self.acceleration = action[1]
        print(f'acceleration = {self.acceleration}')

        self._control.steer = self.steer

        if self.acceleration < 0:
             self._control.brake = np.abs(self.acceleration)
             self._control.throttle = 0

        else:
            self._control.throttle = self.acceleration
            self._control.brake = 0

        print(self._control)    

        self.player.apply_control(self._control)
        self.control_count += 1


    def print_waypoints(self, waypoints):

        for z in waypoints:
            spawn_location_r = carla.Location()
            spawn_location_r.x = float(z[0])
            spawn_location_r.y = float(z[1])
            spawn_location_r.z = 1.0
            self.world.debug.draw_string(spawn_location_r, 'O', draw_shadow=False,
                                                color=carla.Color(r=255, g=0, b=0), life_time=0.1,
                                                persistent_lines=True)
            



    def create_actors(self):

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.filter('*vehicle*')
        self.walker_blueprint = self.blueprint_library.filter('*walker.*')

        # PLAYER
    
        spawn_location = carla.Location()
        spawn_location.x = float(self.args.spawn_x)
        spawn_location.y = float(self.args.spawn_y)
        self.spawn_waypoint = self.map.get_waypoint(spawn_location)
        spawn_transform = self.spawn_waypoint.transform
        spawn_transform.location.z = 1.0
        self.player = self.world.try_spawn_actor(self.vehicle_blueprint.filter('model3')[0], spawn_transform)
        self.world.tick()   
        print('vehicle spawned')


        # Turn on position lights
        current_lights = carla.VehicleLightState.NONE
        current_lights |= carla.VehicleLightState.Position
        self.player.set_light_state(carla.VehicleLightState.Position)

        # CAMERA RGB

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{640}")
        self.rgb_cam.set_attribute("image_size_y", f"{480}")
        self.rgb_cam.set_attribute("fov", f"110")
        self.camera_rgb = self.world.spawn_actor(
            self.rgb_cam,
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(0,0,0)),
            attach_to=self.player)
        self.world.tick()

        # LANE SENSOR

        self.lane_invasion = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'), 
            carla.Transform(), 
            attach_to=self.player)
        self.world.tick()

        # COLLISION SENSOR

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.player)
        self.world.tick()
        
        # SYNCH MODE CONTEXT

        self.synch_mode = CarlaSyncMode(self.world, self.camera_rgb, self.lane_invasion, self.collision_sensor)

        # STATIONARY CAR

        parking_position = carla.Transform(self.player.get_transform().location + carla.Location(-0.5, self.distance_parked, 0.5), 
                                carla.Rotation(0,90,0))
        self.parked_vehicle = self.world.spawn_actor(self.vehicle_blueprint.filter('model3')[0], parking_position)
        self.world.tick()

        # SPECTATOR

        spectator = self.world.get_spectator()
        if self.parked_vehicle is not None:
            transform = self.parked_vehicle.get_transform()
        else:
            transform = self.player.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(y=-10,z=28.5), carla.Rotation(pitch=-90)))
        self.world.tick()

        # CONTROLLER

        self.control_count = 0
        if self.control_mode == "PID":
            self.controller = PIDController.Controller()



    def get_observation(self):

         # EGO information
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_roration = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y
        current_yaw = wrap_angle(current_roration.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)

        current_steer = self.steer


        acceleration_vec =  self.player.get_acceleration()
        current_acceleration = math.sqrt(acceleration_vec.x**2 + acceleration_vec.y**2 + acceleration_vec.z**2)
        sideslip = np.tanh(velocity_vec.x/np.abs(velocity_vec.y+0.1))

  

        self.save_list.append([self.episode_counter,  self.desired_speed, self.last_v, self.ttc_trigger, self.distance_parked, self.clock.get_time(), current_x, current_y, current_speed, current_acceleration, 
                               acceleration_vec.x, acceleration_vec.y, sideslip, current_yaw, current_steer])
