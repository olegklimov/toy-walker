import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import pyglet
from pyglet.gl import *

# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
GAMMA  = 0.99

MOTORS_TORQUE = 80
SPEED_HIP     = 8
SPEED_KNEE    = 12
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 10*SCALE

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE
MAX_TARG_STEP = 64/SCALE

VIEWPORT_W = 800
VIEWPORT_H = 600

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 13    # low long are grass spots, in steps
FRICTION = 2.5
HULL_HEIGHT_POTENTIAL = 5.0  # standing straight .. legs maximum to the sides = ~60 units of distance vertically, to reward using this coef
HULL_ANGLE_POTENTIAL  = 5.0  # keep head level
LEG_POTENTIAL         = 1.0
SPEED_HINT            =  0.005
REWARD_CRASH          = -10.0
REWARD_STOP_PER_FRAME = 1.0

verbose = 1
def log(msg):
    if verbose:
        print(msg)

def leg_targeting_potential(x, y):
    '''
    x - horizontal difference from target
    y - vertical
    https://academo.org/demos/3d-surface-plotter/?expression=(exp(-x%5E2)-0.5)%2F(1%2By)%2B0.02*(x-abs(x))&xRange=-5%2C%2B5&yRange=0%2C%2B10&resolution=100
    '''
    y = max(0,y)
    scale = 1/(0.10*MAX_TARG_STEP)
    x *= scale
    y *= scale
    softabs = 1 + 2*np.log(2) - np.log(1+np.exp(x)) - np.log(1+np.exp(-x))   # maximum 1.0 at x=0, O(x) at +inf, O(-x) at -inf
    # https://www.wolframalpha.com/input/?i=1+%2B+2*ln(2)+-+ln(1%2Bexp(x))+-+ln(1%2Bexp(-x))
    softsign = np.tanh(softabs)   # positive x = -1.6 .. 1.6
    y_step = softsign / (1+y)  # step of 1, positive at x near zero, negative at x outside -1.6 .. 1.6
    return softabs + 2*y_step  # outside target zone, pulling leg up (make y higher) is twice as good as moving towards x=0
    #(np.exp(-(x*scale)**2)-0.5) / (1+y*scale) - 0.55*scale*max(0,abs(x)-1)

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
            self.env.game_over = True
        for leg in self.env.legs:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact += 1
    def EndContact(self, contact):
        for leg in self.env.legs:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact -= 1

class LidarCallback(Box2D.b2.rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & 1) == 0:
            return 1
        self.p2 = point
        self.fraction = fraction
        return 0

class RewardChart:
    def __init__(self):
        self.history = [0.0]
    def push(self, r):
        self.history.append(self.history[-1] + r)
        if len(self.history) > 200:
            self.history.pop(0)
    def draw(self, viewer, pos_x, pos_y, color):
        viewer.draw_polyline( [(
            viewer.scroll + h/SCALE + pos_x*VIEWPORT_H/SCALE,
            pos_y*VIEWPORT_H/SCALE
            ) for h in [0,200]], color=(0.3,0.3,0.3), linewidth=2)
        d = self.history[0]
        viewer.draw_polyline( [(
            viewer.scroll + h/SCALE + pos_x*VIEWPORT_H/SCALE,
            pos_y*VIEWPORT_H/SCALE + 10*(self.history[h]-d)/SCALE
            ) for h in range(len(self.history))], color=color, linewidth=2)

class CommandWalker(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        self._seed()
        self.viewer = None
        self.manual = False
        self.manual_height = 0

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None
        self._reset()

        high = np.array([np.inf]*39)
        self.action_space = spaces.Box(np.array([-1,-1,-1,-1]), np.array([+1,+1,+1,+1]))
        self.observation_space = spaces.Box(-high, high)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.leg_parts:
            self.world.DestroyBody(leg)
        self.leg_parts = []
        self.legs = []
        self.joints = []

    def _generate_terrain(self, hardcore):
        GRASS, STAIRS, STUMP, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        startpad = 0
        counter  = 1
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        self.init_x_good = 65535
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                velocity += 2*self.np_random.uniform(-1, 1)/SCALE
                y += velocity
                if counter==startpad and self.init_x_good > np.abs(i - TERRAIN_LENGTH//2):
                    self.init_x_good = np.abs(i - TERRAIN_LENGTH//2)
                    self.init_x = x
                    self.init_y = y

            elif state==PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-4*TERRAIN_STEP),
                    (x,              y-4*TERRAIN_STEP),
                    ]
                t = self.world.CreateStaticBody(
                    fixtures = fixtureDef(
                        shape=polygonShape(vertices=poly),
                        friction = FRICTION
                    ))
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                t = self.world.CreateStaticBody(
                    fixtures = fixtureDef(
                        shape=polygonShape(vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]),
                        friction = FRICTION
                    ))
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                t = self.world.CreateStaticBody(
                    fixtures = fixtureDef(
                        shape=polygonShape(vertices=poly),
                        friction = FRICTION
                    ))
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    t = self.world.CreateStaticBody(
                        fixtures = fixtureDef(
                            shape=polygonShape(vertices=poly),
                            friction = FRICTION
                        ))
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - 2 # - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height-0.5)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS and hardcore:
                    state = self.np_random.randint(GRASS+1, _STATES_)
                    oneshot = True
                elif state==GRASS and not hardcore:
                    state = self.np_random.randint(GRASS, STAIRS+1)  # grass or stairs
                    oneshot = True
                else:
                    state = GRASS
                    startpad = counter//2
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            t = self.world.CreateStaticBody(
                fixtures = fixtureDef(
                    shape=edgeShape(vertices=poly),
                    friction = FRICTION,
                    categoryBits=0x0001,
                ))
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def _reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.ts = 0
        self.scroll = 0.0
        self.lidar_render = 0
        self.chart_legs   = RewardChart()
        self.chart_height = RewardChart()
        self.chart_angle = RewardChart()
        self.chart_misc  = RewardChart()
        self.external_command = 0
        self.manual_jump = 0
        self.steps_done = 0
        self.jump = 0
        self.hull_desired_position = self.np_random.uniform(low=-1, high=+1)

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = self.init_x
        init_y = self.init_y + 2.1*LEG_H
        self.hull = self.world.CreateDynamicBody(
            position = (init_x, init_y),
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[ (x/SCALE,y/SCALE) for x,y in HULL_POLY ]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0020,
                maskBits=0x001,  # collide only with ground
                restitution=0.0) # 0.99 bouncy
                )
        self.hull.color1 = (0.5,0.4,0.9)
        self.hull.color2 = (0.3,0.3,0.5)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)
        self.hull.ApplyTorque(self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), True);

        self.legs = []
        self.leg_parts = []
        self.joints = []
        for i in [-1,+1]:
            leg = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/2, LEG_H/2)),
                    density=10.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.color1 = (0.6+i/10., 0.3+i/10., 0.5+i/10.)
            leg.color2 = (0.4+i/10., 0.2+i/10., 0.3+i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -0.9,
                upperAngle = 1.2,
                )
            self.leg_parts.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position = (init_x, init_y - LEG_H*3/2 - LEG_DOWN),
                angle = (i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(0.8*LEG_W/2, LEG_H/2)),
                    density=10.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            lower.color1 = leg.color1
            lower.color2 = leg.color2
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.9,
                upperAngle = -0.1,
                )
            lower.ground_contact = 0
            self.leg_parts.append(lower)
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.leg_parts.reverse()
        self.drawlist = self.terrain + self.leg_parts + [self.hull]

        self.lidar = [LidarCallback() for _ in range(20)]

        self.target = np.zeros( (2,) )
        self.hill_x = np.zeros( (2,) )
        self.hill_y = np.zeros( (2,) )
        self.leg_active = 0
        self.legs[0].tip_x = 0  # dummy values
        self.legs[0].tip_y = 0
        self.legs[1].tip_x = 0
        self.legs[1].tip_y = 0
        self.world.Step(1.0/FPS, 6*30, 2*30)
        _, self.potential_height, self.potential_angle = self._potentials()  # used here, first result ignored

        return self._step(np.array([0,0,0,0]))[0]

    def _step(self, action):
        #self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP  * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP  * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed     = float(SPEED_HIP     * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
            self.joints[1].motorSpeed     = float(SPEED_KNEE    * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
            self.joints[2].motorSpeed     = float(SPEED_HIP     * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
            self.joints[3].motorSpeed     = float(SPEED_KNEE    * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.ts += 1

        for leg in self.legs:
            leg.tip_x = leg.position[0] + np.sin(leg.angle) * 0.5*LEG_H
            leg.tip_y = leg.position[1] - np.cos(leg.angle) * 0.5*LEG_H
        if self.manual: self.hull_desired_position = self.manual_height

        self._set_feet_target()

        ######################################################## REWARDS ####################################################################
        LEAK = 1-GAMMA
        potential_legs, potential_height, potential_angle = self._potentials()

        reward_legs   = potential_legs   - self.potential_legs
        reward_height = potential_height - self.potential_height
        reward_angle  = potential_angle  - self.potential_angle
        self.potential_legs   = potential_legs
        self.potential_height = potential_height
        self.potential_angle  = potential_angle

        #reward_stop_alive = 0
        reward_leg_hint = 0
        if self.external_command==0:
            if self.legs[0].ground_contact and self.legs[1].ground_contact:
                #reward_stop_alive = REWARD_STOP_PER_FRAME - np.abs(self.hull.linearVelocity.x*SCALE/FPS)
                potential_height = max(0, potential_height-np.abs(self.hull.linearVelocity.x*SCALE/FPS))
            else:
                potential_height = 0
            #log("STOP ALIVE REWARD %0.2f" % potential_height)
        elif self.external_command in [-1,+1]:
            prop_leg = self.legs[1-self.leg_active]
            prop_dist = prop_leg.tip_x - self.hull.position[0]
            if prop_dist*self.external_command > 0 and prop_leg.ground_contact:
                reward_leg_hint = SPEED_HINT*SCALE*self.external_command*self.hull.linearVelocity.x*SCALE/FPS
            else:
                reward_leg_hint = 0
            #log("LEG HINT REWARD %0.2f" % reward_leg_hint)

        reward_jump = 0
        #if self.jump > 0:
        #    self.jump -= 1
        #    reward_jump = SPEED_HINT*SCALE*max(0, self.hull.linearVelocity.y)*SCALE/FPS
        #    log("JUMP REWARD %0.2f" % reward_jump)

        reward  = reward_legs + reward_height + reward_angle
        reward += LEAK*potential_height + LEAK*potential_angle
        reward += reward_leg_hint + reward_jump
        self.chart_legs.push(reward_legs)
        self.chart_height.push(reward_height + LEAK*potential_height)
        self.chart_angle.push(reward_angle + LEAK*potential_angle)
        self.chart_misc.push(reward_leg_hint)
        #log("potential_legs %8.2f (%+8.2f)   potential_height %8.2f (%+8.2f)  potential_angle %8.2f (%+8.2f)" %
        #    (potential_legs,reward_legs, potential_height,reward_height, potential_angle,reward_angle))

        #####################################################################################################################################

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(20):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*(i-9.5)/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*(i-9.5)/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            vel.x*300/SCALE/FPS,  # Normalized to get -1..1 range
            vel.y*300/SCALE/FPS,
            self.joints[0].angle,   # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[0].ground_contact>0 else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact>0 else 0.0,
            (self.target[0] - self.hull.position[0]) / MAX_TARG_STEP,
            (self.target[1] - self.hull.position[0]) / MAX_TARG_STEP,
            self.external_command,
            self.hull_desired_position,
            1 if self.jump>0 else 0
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==39

        self.scroll = pos.x - VIEWPORT_W/SCALE/2

        done = False
        if self.game_over or pos[0] < 0:
            reward = REWARD_CRASH
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            done   = True
        return np.array(state), reward, done, {}

    def command(self, new_command):
        if self.external_command==new_command: return
        if new_command in [+1,+2]: self.leg_active = (1 if self.target[1] < self.target[0] else 0)  # back leg active
        if new_command in [-1,-2]: self.leg_active = (1 if self.target[1] > self.target[0] else 0)
        if new_command in [0]: self.target[0] = self.target[1] = 0
        log("COMMAND %+i -> %+i, active=%i" % (self.external_command, new_command, self.leg_active))
        self.external_command = new_command

    def _set_feet_target(self):
        legs = self.legs
        targ = self.target
        hill = self.hill_x

        reset_potential = False
        allow_change_command = 0.0
        allow_jump = 0.0
        if self.external_command in [0]:
            allow_change_command = 0.05 if legs[0].ground_contact and legs[1].ground_contact else 0.0
            allow_jump = 0.01 if legs[0].ground_contact and legs[1].ground_contact else 0.0
        if self.external_command in [-1,+1]:
            allow_change_command = 0.01 if self.steps_done >= 2 else 0.0
            allow_jump = 0.01 if legs[0].ground_contact or legs[1].ground_contact else 0.0
        allow_jump = 0.0
        if self.np_random.rand() < allow_change_command and not self.manual:
            while 1:
                new_command = self.np_random.randint(low=0, high=+2)
                if self.external_command==new_command: continue
                break
            self.command(new_command)
            reset_potential = True
        a = self.leg_active
        if (self.np_random.rand() < allow_jump and not self.manual) or self.manual_jump==1:
            self.manual_jump = 0
            self.jump = 10

        if targ[0]==0 and targ[1]==0: # initial
            diff = MAX_TARG_STEP*self.np_random.uniform(0.3, 0.5)
            targ[0] = self.hull.position[0] - diff
            targ[1] = self.hull.position[0] + diff
            if self.np_random.rand() > 0.5: targ[0], targ[1] = targ[1], targ[0]
            hill[0] = targ[0]
            hill[1] = targ[1]
            reset_potential = True
            assert(self.external_command==0)  # starts in the air

        elif self.external_command==0:
            self.steps_done = 0
            diff = (hill[1] - hill[0]) / 2
            hill[0] = targ[0] = self.hull.position[0] - diff
            hill[1] = targ[1] = self.hull.position[0] + diff

        elif self.external_command==+1:
            assert(targ[a] < targ[1-a]) # loop invariant for walking
            if targ[a] < hill[1-a]:  # here from STOP command (targ[a] was on left, hill[1-a] stays the same to compare correctly)
                log("DETECTED +1 FROM STOP")
                hill[1-a] = legs[1-a].tip_x
                targ[a]   = legs[1-a].tip_x + MAX_TARG_STEP*self.np_random.uniform(0.5, 1)
                hill[a]   = targ[a]
                targ[1-a] = targ[a] + MAX_TARG_STEP*self.np_random.uniform(0.5, 1)
                reset_potential = True
            if legs[a].ground_contact and legs[a].tip_x > legs[1-a].tip_x: # step made
                self.steps_done += 1
                log("STEP +1 SWITCH LEGS")
                hill[a]   = legs[a].tip_x # de-facto leg placement, turn into potential well
                targ[1-a] = np.clip(targ[1-a], legs[a].tip_x + 0.5*MAX_TARG_STEP, legs[a].tip_x + MAX_TARG_STEP) # long-visible target becomes gravity well
                hill[1-a] = targ[1-a]
                targ[a]   = targ[1-a] + MAX_TARG_STEP*self.np_random.uniform(0.5, 1)
                a = 1-a
                reset_potential = True
            allow_random_command = 0.01

        elif self.external_command==-1:
            assert(targ[a] > targ[1-a])
            if targ[a] > hill[1-a]:
                log("DETECTED -1 FROM STOP")
                hill[1-a] = legs[1-a].tip_x
                targ[a]   = legs[1-a].tip_x - MAX_TARG_STEP*self.np_random.uniform(0.5, 1)
                hill[a]   = targ[a]
                targ[1-a] = targ[a] - MAX_TARG_STEP*self.np_random.uniform(0.5, 1)
                reset_potential = True
            if legs[a].ground_contact and legs[a].tip_x < legs[1-a].tip_x:
                self.steps_done += 1
                log("STEP -1 SWITCH LEGS")
                hill[a]   = legs[a].tip_x
                targ[1-a] = np.clip(targ[1-a], legs[a].tip_x - MAX_TARG_STEP, legs[a].tip_x - 0.5*MAX_TARG_STEP)
                hill[1-a] = targ[1-a]
                targ[a]   = targ[1-a] - MAX_TARG_STEP*self.np_random.uniform(0.5, 1)
                a = 1-a
                reset_potential = True
            allow_random_command = 0.01

        lidar = LidarCallback()
        for i in [0,1]:
            lidar.fraction = 1.0
            lidar.p1 = (self.hill_x[i], +1000)
            lidar.p2 = (self.hill_x[i], -1000)
            self.world.RayCast(lidar, lidar.p1, lidar.p2)
            self.hill_y[i] = lidar.p2[1]

        if reset_potential: # without feeding to reward (would be spike)
            self.potential_legs, _, _ = self._potentials()
        self.leg_active = a

    def _potentials(self):
        self.legs_level = min(self.legs[0].position[1], self.legs[1].position[1])
        above1 = 1.30  # for hull_desired_position==-1
        above2 = 1.75  # for hull_desired_position==+1
        above = above1 + (above2-above1)*((self.hull_desired_position+1)/2)
        self.hull_above_legs_shouldbe = self.legs_level + above*LEG_H
        potential_height = (self.hull.position[1] - self.hull_above_legs_shouldbe) / LEG_H
        self.hull_dangerous_level = (self.hull.position[1] - self.legs_level) / LEG_H
        #log("potential_height", potential_height, "above", above)
        if self.hull_dangerous_level < 1.15: self.game_over = True

        leg0_pot = leg_targeting_potential(self.legs[0].tip_x - self.hill_x[0], self.legs[0].tip_y - self.hill_y[0])
        leg1_pot = leg_targeting_potential(self.legs[1].tip_x - self.hill_x[1], self.legs[1].tip_y - self.hill_y[1])
        #print("pot[a]   = %0.2f" % leg0_pot if self.leg_active else leg1_pot)
        #print("pot[1-a] = %0.2f" % leg1_pot if self.leg_active else leg0_pot)
        if self.external_command in [+1,-1]:
            pass
        else:
            leg0_pot *= 0.3  # stop: legs less important, stays here only as a tip
            leg1_pot *= 0.3

        return (
            LEG_POTENTIAL*leg0_pot + LEG_POTENTIAL*leg1_pot,
            HULL_HEIGHT_POTENTIAL*np.exp( -np.square(potential_height)/(2*np.square(0.15)) ),  # 0.15 of leg height is acceptable range
            HULL_ANGLE_POTENTIAL*np.exp( -np.square(self.hull.angle)/(2*np.square(0.1)) ),
            )

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render+1) % 100
        i = self.lidar_render
        if i < 2*len(self.lidar):
            l = self.lidar[i] if i < len(self.lidar) else self.lidar[len(self.lidar)-i-1]
            self.viewer.draw_polyline( [l.p1, l.p2], color=(1,0,0), linewidth=1 )

        color = (0.8,0.8,0.8)
        if self.hull_dangerous_level < 1.20: color = (1,0,0)
        x = self.hull.position[0]
        y = self.hull_above_legs_shouldbe
        self.viewer.draw_polyline( [(x+dx,y) for dx in [-2*MAX_TARG_STEP,+2*MAX_TARG_STEP]], color=color, linewidth=1 )
        y = self.legs_level
        self.viewer.draw_polyline( [(x+dx,y) for dx in [-2*MAX_TARG_STEP,+2*MAX_TARG_STEP]], color=color, linewidth=1 )

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        for leg in self.legs:
            if leg.ground_contact > 0:
                t = rendering.Transform(translation=(leg.tip_x,leg.tip_y))
                color = (0,0,1)
                self.viewer.draw_circle(4/SCALE, 10, color=color).add_attr(t)
            if leg==self.legs[self.leg_active]:
                t = rendering.Transform(translation=leg.position)
                self.viewer.draw_circle(2/SCALE, 10, color=(1,0,0)).add_attr(t)

        for i in [0,1]:
            hill_x = self.hill_x[i]
            target_x = self.target[i]
            hill_y = self.hill_y[i]
            color = self.legs[i].color1
            self.viewer.draw_polyline( [(
                hill_x + dx,
                hill_y + 0.5*leg_targeting_potential(dx, 0),
                ) for dx in np.arange(-2*MAX_TARG_STEP,+2*MAX_TARG_STEP,MAX_TARG_STEP/15)], color=color, linewidth=1)
            self.viewer.draw_polyline( [(
                hill_x + dx,
                hill_y + 0.5*leg_targeting_potential(dx, 1) + 1/SCALE,
                ) for dx in np.arange(-2*MAX_TARG_STEP,+2*MAX_TARG_STEP,MAX_TARG_STEP/15)], color=color, linewidth=1)
            t = rendering.Transform(translation=(target_x, hill_y))
            self.viewer.draw_circle(5/SCALE, 10, color=color).add_attr(t)

        self.viewer.scroll = self.scroll
        self.chart_legs.draw(self.viewer,   0.4, 0.6, (1,  0,  0))
        self.chart_height.draw(self.viewer, 0.7, 0.6, (0,  0.5,0))
        self.chart_angle.draw(self.viewer,  1.0, 0.6, (0,  0,  0.5))
        self.chart_misc.draw(self.viewer,   0.4, 0.4, (0.5,0.3,0.3))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def print_state(self, s):
        if not verbose: return
        log("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
        log("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
        log("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
        log("targ " + str(["{:+0.2f}".format(x) for x in s[14:16]]) + " command=%0.2f height %0.2f jump %0.2f" % (s[16], s[17], s[18]))

class CommandWalkerHardcore(CommandWalker):
    hardcore = True

def heuristic(env, s):
    # Heurisic: suboptimal, have no notion of balance.
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    SPEED = 0.28  # Will fall forward on higher speed
    SUPPORT_KNEE_ANGLE = +0.1

    if "state" not in env.__dict__: # Init
        env.state = STAY_ON_ONE_LEG
        env.moving_leg = 0
        env.supporting_knee_angle = SUPPORT_KNEE_ANGLE

    a = np.array([0.0, 0.0, 0.0, 0.0])
    supporting_leg = 1 - env.moving_leg

    contact0 = s[8]
    contact1 = s[13]
    moving_s_base = 4 + 5*env.moving_leg
    supporting_s_base = 4 + 5*supporting_leg

    hip_targ  = [None,None]   # -0.8 .. +1.1
    knee_targ = [None,None]   # -0.6 .. +0.9
    hip_todo  = [0.0, 0.0]
    knee_todo = [0.0, 0.0]

    if env.state==STAY_ON_ONE_LEG:
        hip_targ[env.moving_leg]  = 1.1
        knee_targ[env.moving_leg] = -0.6
        env.supporting_knee_angle += 0.03
        if s[2] > SPEED: env.supporting_knee_angle += 0.03
        env.supporting_knee_angle = min( env.supporting_knee_angle, SUPPORT_KNEE_ANGLE )
        knee_targ[supporting_leg] = env.supporting_knee_angle
        if s[supporting_s_base+0] < 0.10: # supporting leg is behind
            env.state = PUT_OTHER_DOWN
    if env.state==PUT_OTHER_DOWN:
        hip_targ[env.moving_leg]  = +0.1
        knee_targ[env.moving_leg] = SUPPORT_KNEE_ANGLE
        knee_targ[supporting_leg] = env.supporting_knee_angle
        if s[moving_s_base+4]:
            env.state = PUSH_OFF
            env.supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
    if env.state==PUSH_OFF:
        knee_targ[env.moving_leg] = env.supporting_knee_angle
        knee_targ[supporting_leg] = +1.0
        if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
            env.state = STAY_ON_ONE_LEG
            env.moving_leg = 1 - env.moving_leg
            supporting_leg = 1 - env.moving_leg

    if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
    if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
    if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
    if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

    hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
    hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
    knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
    knee_todo[1] -= 15.0*s[3]

    a[0] = hip_todo[0]
    a[1] = knee_todo[0]
    a[2] = hip_todo[1]
    a[3] = knee_todo[1]
    a = np.clip(0.5*a, -1.0, 1.0)
    return a

if __name__=="__main__":
    env = CommandWalker()
    s = env.reset()
    steps = 0
    total_reward = 0
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 1 == 0 or done:
            log("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            log("step {} total_reward {:+0.2f}".format(steps, total_reward))
            log("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            log("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            log("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
            log("targ " + str(["{:+0.2f}".format(x) for x in s[14:16]]))
        steps += 1
        env.render()
        if done: break
