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

MOTORS_TORQUE = 80
SPEED_HIP     = 4
SPEED_KNEE    = 6
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

HULL_POLY =[
    (-30,+9), (+6,+9), (+34,+1),
    (+34,-8), (-30,-8)
    ]
LEG_DOWN = -8/SCALE
LEG_W, LEG_H = 8/SCALE, 34/SCALE
MAX_TARG_STEP = 54/SCALE

VIEWPORT_W = 3600
VIEWPORT_H = 500

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 10    # in steps
FRICTION = 2.5
REWARD_STEP = 10
HULL_HEIGHT_POTENTIAL = 1.0  # standing straight .. legs maximum to the sides = ~60 units of distance vertically, to reward using this coef
HULL_ANGLE_POTENTIAL  = 1.0  # keep head level
LEG_POTENTIAL         = 1.0  # angle in radians, about -0.8..1.1,

def leg_targeting_potential(x, y):
    '''
    x - horizontal difference from target
    y - vertical
    https://academo.org/demos/3d-surface-plotter/?expression=(exp(-x%5E2)-0.5)%2F(1%2By)%2B0.02*(x-abs(x))&xRange=-5%2C%2B5&yRange=0%2C%2B10&resolution=100
    '''
    y = max(0,y)
    scale = 1/(0.2*MAX_TARG_STEP)
    return (np.exp(-(x*scale)**2)-0.5) / (1+y*scale) + 0.02*scale*(x-abs(x))

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

class CommandWalker(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    hardcore = False

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None

        self.prev_shaping = None
        self._reset()

        high = np.array([np.inf]*26)
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
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += 2*self.np_random.uniform(-1, 1)/SCALE
                y += velocity

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
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
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
        self.reward_height = 0.0
        self.reward_legs = 0.0
        self.reward_history = []
        if self.np_random.randint(low=0, high=+2)==1:
            self.external_command = +1
        else:
            self.external_command = -1
        self.hull_desired_position = 1.45*LEG_H
        #print("self.external_command", self.external_command)

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP*TERRAIN_STARTPAD/2
        init_y = TERRAIN_HEIGHT+2.1*LEG_H
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
            leg.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            leg.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = i,
                lowerAngle = -0.8,
                upperAngle = 1.1,
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
            lower.color1 = (0.6-i/10., 0.3-i/10., 0.5-i/10.)
            lower.color2 = (0.4-i/10., 0.2-i/10., 0.3-i/10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H/2),
                localAnchorB=(0, LEG_H/2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed = 1,
                lowerAngle = -1.6,
                upperAngle = -0.1,
                )
            lower.ground_contact = 0
            self.leg_parts.append(lower)
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.leg_parts + [self.hull]

        self.lidar = [LidarCallback() for _ in range(10)]

        self.target = np.zeros( (2,) )
        self.target_y = np.zeros( (2,) )
        self._set_feet_target()
        self.potential_hull, self.potential_legs = self._potentials()

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

        was_leg0_contact = self.legs[0].ground_contact > 0
        was_leg1_contact = self.legs[1].ground_contact > 0

        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.ts += 1

        for leg in self.legs:
            leg.tip_x = leg.position[0] + np.sin(leg.angle) * 0.5*LEG_H
            leg.tip_y = leg.position[1] - np.cos(leg.angle) * 0.5*LEG_H

        if was_leg0_contact==False and self.legs[0].ground_contact > 0 or was_leg1_contact==False and self.legs[1].ground_contact > 0:
            self._set_feet_target()

        potential_hull, potential_legs = self._potentials()

        reward_height = potential_hull - self.potential_hull
        reward_legs = potential_legs - self.potential_legs
        self.potential_hull = potential_hull
        self.potential_legs = potential_legs
        self.reward_legs += reward_legs
        #####################################################################################################################################
        print("potential_legs %8.2f (%+8.2f)   potential_hull %8.2f (%+8.2f)" % (potential_legs, reward_legs, potential_hull, reward_height))

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5*i/10.0)*LIDAR_RANGE,
                pos[1] - math.cos(1.5*i/10.0)*LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,        # Normal angles up to 0.5 here, but sure more is possible.
            2.0*self.hull.angularVelocity/FPS,
            0.3*vel.x*(VIEWPORT_W/SCALE)/FPS,  # Normalized to get -1..1 range
            0.3*vel.y*(VIEWPORT_H/SCALE)/FPS,
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
            ]
        state += [l.fraction for l in self.lidar]
        assert len(state)==26

        self.scroll = pos.x - VIEWPORT_W/SCALE/2

        reward = reward_legs + reward_height
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        done = False
        if self.game_over or pos[0] < 0:
            reward = -1
            done   = True
        if pos[0] > (TERRAIN_LENGTH-TERRAIN_GRASS)*TERRAIN_STEP:
            print("STAT reward_legs        = %0.2f" % self.reward_legs)
            print("STAT reward_height      = %0.2f" % self.reward_height)
            done   = True
        return np.array(state), reward, done, {}

    def _set_feet_target(self):
        # want forward leg first
        legs = self.legs[:]
        targ = self.target[:]
        swap = legs[1].position[0] > legs[0].position[0]
        if swap:
            legs[0], legs[1] = legs[1], legs[0]
            targ[0], targ[1] = targ[1], targ[0]

        reset_potential = False

        if targ[0]==0 and targ[1]==0:
            # initial
            fwd = legs[0].position[0]
            targ[1] = fwd     + MAX_TARG_STEP*self.np_random.uniform(0.2, 1)
            targ[0] = targ[1] + MAX_TARG_STEP*self.np_random.uniform(0.2, 1)
            if self.np_random.randint(low=0, high=+2)==1:
                targ[0], targ[1] = targ[1], targ[0]
            self.target_switch_ts = self.ts

        elif legs[0].ground_contact and targ[0] < targ[1]:
            # target fulfilled
            targ[1] = min(targ[1], legs[0].position[0] + MAX_TARG_STEP)      # if stepped short of the target, keep next step within possible limit
            targ[1] = max(targ[1], legs[0].position[0] + 0.2*MAX_TARG_STEP)  # make target forward of leg
            targ[0] = targ[1] + MAX_TARG_STEP*self.np_random.uniform(0.2, 1)
            self.target_switch_ts = self.ts
            reset_potential = True

        if swap:
            self.target[0], self.target[1] = targ[1], targ[0]
        else:
            self.target[0], self.target[1] = targ[0], targ[1]

        lidar = LidarCallback()
        for i in [0,1]:
            lidar.fraction = 1.0
            lidar.p1 = (self.target[i], +1000)
            lidar.p2 = (self.target[i], -1000)
            self.world.RayCast(lidar, lidar.p1, lidar.p2)
            self.target_y[i] = lidar.p2[1]

        if reset_potential: # without feeding to reward (would be spike)
            _, self.potential_legs = self._potentials()

    def _potentials(self):
        self.hull_above_legs = self.hull.position[1] - 0.5*(self.legs[0].position[1] + self.legs[1].position[1])
        potential_hull = - HULL_HEIGHT_POTENTIAL * abs( self.hull_desired_position - self.hull_above_legs ) / LEG_H

        #self.bad_height = above_legs < 1.4*LEG_H       # down on knees, get up!
        #above_legs = min( 0, above_legs - 1.4*LEG_H )  # non-zero and negative only when bad_height

        leg0_pot = 0
        leg1_pot = 0
        if self.target[0] > self.target[1]:
            leg1_pot = leg_targeting_potential(self.legs[1].position[0] - self.target[0], self.legs[1].position[1] - self.target[1])
        else:
            leg0_pot = leg_targeting_potential(self.legs[0].position[0] - self.target[0], self.legs[0].position[1] - self.target[1])

        return (
            potential_hull,
            LEG_POTENTIAL*leg0_pot + LEG_POTENTIAL*leg1_pot - HULL_ANGLE_POTENTIAL*np.abs(self.hull.angle)
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

        # self.hull_above_legs = self.hull.position[1] - 0.5*(self.legs[0].position[1] + self.legs[1].position[1])
        # potential_hull = HULL_HEIGHT_POTENTIAL * abs( self.hull_desired_position - self.hull_above_legs ) / LEG_H
        x = self.hull.position[0]
        y = self.hull_desired_position + 0.5*(self.legs[0].position[1] + self.legs[1].position[1])
        self.viewer.draw_polyline( [(x+dx,y) for dx in [-2*MAX_TARG_STEP,+2*MAX_TARG_STEP]], color=(1,0,0), linewidth=1 )

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

        for i in [0,1]:
            target_x = self.target[i]
            target_y = self.target_y[i]
            color = self.legs[i].color1
            self.viewer.draw_polyline( [(
                target_x + dx,
                target_y + 5*leg_targeting_potential(dx, 0),
                ) for dx in np.arange(-MAX_TARG_STEP,+MAX_TARG_STEP,MAX_TARG_STEP/30)], color=color, linewidth=1)
            t = rendering.Transform(translation=(target_x, target_y))
            self.viewer.draw_circle(5/SCALE, 10, color=color).add_attr(t)

        self.viewer.draw_polyline( [(
            self.scroll + h/SCALE + 0.2*VIEWPORT_H/SCALE,
            0.8*VIEWPORT_H/SCALE
            ) for h in [0,100]], color=(0.3,0.3,0.3), linewidth=2)
        self.viewer.draw_polyline( [(
            self.scroll + h/SCALE + 0.2*VIEWPORT_H/SCALE,
            0.8*VIEWPORT_H/SCALE + self.reward_history[h]/SCALE*100
            ) for h in range(len(self.reward_history))], color=(1,0,0), linewidth=2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

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
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
            print("targ " + str(["{:+0.2f}".format(x) for x in s[14:16]]))
        steps += 1
        env.render()
        if done: break
