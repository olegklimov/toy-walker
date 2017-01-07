import os, sys, subprocess, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tinkerbell.logger as logger
from rl_algs.common.mpi_fork import mpi_fork
from rl_algs import pposgd
#from rl_algs.pposgd.mlp_policy import MlpPolicy
#from rl_algs.sandbox.hoj.common import logx as logger
#import rl_algs.sandbox.oleg.evolution

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
num_cpu = 8

import gym
from gym.envs.registration import register
from gym.wrappers import SkipWrapper
skip_wrap = lambda x: x
#skip_wrap = SkipWrapper(2)

env_id = sys.argv[1]
experiment = sys.argv[2]
max_timesteps = 16000000

if len(sys.argv)>3:
    demo = sys.argv[3]=="demo"
    manual = sys.argv[3]=="manual"
    if not demo and not manual:
        load_previous_experiment = sys.argv[2]
else:
    demo = False
    manual = False
    load_previous_experiment = None

if os.getenv("IN_MPI") is None:
    print("environment:              '%s'" % env_id)
    print("experiment_name:          '%s'" % experiment)
    if not demo and not manual:
        print("load_previous_experiment: '%s'" % load_previous_experiment)
    else:
        print("demo:   %s" % demo)
        print("manual: %s" % manual)

if env_id=='CommandWalker-v0':
    import command_walker
    register(
        id='CommandWalker-v0',
        entry_point='command_walker:CommandWalker',
        timestep_limit=700,
        )


# ------------------------------- network ----------------------------------

policy_kwargs = dict(
    hid_size=196,
    num_hid_layers=2
    )

def policy_fn(name, ob_space, ac_space):
    import rl_algs.common.tf_util as U
    from rl_algs.common.distributions import make_pdtype
    from rl_algs.common.mpi_running_mean_std import RunningMeanStd
    class ModifiedPolicy(object):
        recurrent = False

        def __init__(self, name, *args, **kwargs):
            with tf.variable_scope(name):
                self._init(*args, **kwargs)
                self.scope = tf.get_variable_scope().name

        def _init(self, ob_space, ac_space, hid_size, num_hid_layers):
            assert isinstance(ob_space, gym.spaces.Box)

            self.pdtype = pdtype = make_pdtype(ac_space)
            sequence_length = None

            ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

            # supervised
            #x = tf.nn.relu( U.dense(x, 256, "dense1", weight_init=U.normc_initializer(1.0), learnable=learnable) )

            # value est
            with tf.variable_scope("retfilter"):
                self.ret_rms = RunningMeanStd()
            last_out = ob
            for i in range(num_hid_layers):
                last_out = tf.nn.relu(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            self.vpredz = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
            self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean # raw = not standardized

            # action
            last_out = ob
            for i in range(num_hid_layers):
                last_out = tf.nn.relu(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            if isinstance(ac_space, gym.spaces.Box):
                mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer)
                #pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
                pdparam = U.concatenate([ 2.0*tf.nn.tanh(mean), mean * 0.0 - 0.2 + 0.0*logstd ], axis=1)
                # -0.2 => 0.8 (works for ppo/walking)
                # -0.5 => 0.6
                # -1.6 => 0.2 (works for evolution)
                # -2.3 => 0.1
            else:
                bug()
                pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(pdparam)

            # ---
            self.state_in = []
            self.state_out = []

            stochastic = tf.placeholder(dtype=tf.bool, shape=())
            ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
            self._act = U.function([stochastic, ob], [ac, self.vpred])  #, logstd

        def act(self, stochastic, ob):
            ac1, vpred1 =  self._act(stochastic, ob[None])
            #print(std)
            return ac1[0], vpred1[0]
        def get_variables(self):
            return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
        def get_trainable_variables(self):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        def get_initial_state(self):
            return []

    return ModifiedPolicy(name=name, ob_space=ob_space, ac_space=ac_space, **policy_kwargs)


# ------------------------- learn -----------------------------

if not demo and not manual:
    def train():
        learn_kwargs = dict(
            timesteps_per_batch=1024, # horizon
            max_kl=0.03, clip_param=0.2, entcoeff=0.00, # objective
            #klcoeff=0.01, adapt_kl=0,
            klcoeff=0.001, adapt_kl=0.03,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64, linesearch=True, # optimization
            gamma=0.99, lam=0.95, # advantage estimation
            )
        # optim_epochs 24 => good
        # optim_epochs 10 => slightly worse, but faster
        # batch 16  => too slow
        # batch 64  most experiments
        # batch 128 => can't converge
        whoami = mpi_fork(num_cpu)
        if whoami=="parent":
            return
        env = gym.make(env_id)
        if env_id=='CommandWalker-v0':
            command_walker.verbose = 0
            env.experiment(experiment, False)
        env = skip_wrap(env)

        import rl_algs.common.tf_util as U

        progress_dir = os.path.join(env_id, "progress")
        os.makedirs(os.path.join(progress_dir, "monitor"), exist_ok=True)
        logger.set_expt_dir(progress_dir)
        logger.remove_text_output(sys.stdout)
        if rank==0:
            tab_fn = "%s.csv" % (experiment)
            log_fn = "%s.log" % (experiment)
            with open(os.path.join(progress_dir, log_fn), "w"): pass
            with open(os.path.join(progress_dir, tab_fn), "w"): pass
            logger.add_tabular_output(tab_fn)
            logger.add_text_output(log_fn)
            logger.info(subprocess.Popen(['git', 'diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout.read())

        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            device_count = { "GPU": 0 },
            )
        config.gpu_options.per_process_gpu_memory_fraction = 0.07
        sess = tf.InteractiveSession(config=config)

        class Dummy:
            pass
        dummy = Dummy()
        dummy.timesteps_so_far = 0
        EVERY = 100000

        def save_policy(pi, timesteps_so_far):
            if timesteps_so_far > dummy.timesteps_so_far + EVERY:
                dummy.timesteps_so_far += EVERY
            else:
                return
            tv_list = pi.get_trainable_variables()
            print("SAVE")
            print("tainable:")
            for v in tv_list:
                print("\t'%s'" % v.name)
            saver = tf.train.Saver(var_list=tv_list)
            saver.save(sess, 'models/%s' % experiment)

        def load_policy(pi):
            if not load_previous_experiment: return
            print("LOAD '%s'" % load_previous_experiment)
            tv_list = pi.get_trainable_variables()
            saver = tf.train.Saver(var_list=tv_list)
            saver.restore(sess, 'models/%s' % load_previous_experiment)
            sys.stdout.flush()

        env.monitor.start(os.path.join(logger.get_expt_dir(), "monitor"), force=True, video_callable=False)
        if rank==0:
            learn_kwargs["save_callback"] = save_policy
        learn_kwargs["load_callback"] = load_policy

        if 1:
            pposgd.learn(env, policy_fn,
                max_timesteps=max_timesteps,
                **learn_kwargs)
        else:
            learn_kwargs["gamma"] = 0.995
            rl_algs.sandbox.oleg.evolution.learn(env, policy_fn,
                max_timesteps=max_timesteps,
                episodes_per_batch_per_worker=10,
                perturbation=0.05,
                **learn_kwargs)

        env.monitor.close()

    train()


# ------------------------- demo -----------------------------

else:
    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.InteractiveSession(config=config)

    env = gym.make(env_id)
    if env_id=='CommandWalker-v0':
        command_walker.verbose = 0
        env.experiment(experiment, playback=True)
    env.manual = manual
    env = skip_wrap(env)
    #env.monitor.start("demo", force=True)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    tf.global_variables_initializer().run()

    print("Loading '%s'" % experiment)
    tv_list = pi.get_trainable_variables()
    saver = tf.train.Saver(var_list=tv_list)

    human_sets_pause = False
    from pyglet.window import key as kk
    keys = {}
    def key_event(pressed, key, mod):
        keys[key] = +1 if pressed else 0
        global human_sets_pause, human_wants_restart
        if pressed and key==kk.SPACE: human_sets_pause = not human_sets_pause
        if pressed and key==0xff0d: human_wants_restart = True
        if pressed and key==ord('q'): sys.exit(0)
        if env_id!='CommandWalker-v0': return
        command = keys.get(kk.RIGHT, 0) - keys.get(kk.LEFT, 0)
        env.command(command)
        env.manual_jump = keys.get(kk.LSHIFT, 0)
        env.manual_height = keys.get(kk.UP, 0) - keys.get(kk.DOWN, 0)
        #print("(%x key=%x)" % (mod, key))
    def key_press(key, mod): key_event(True, key, mod)
    def key_release(key, mod): key_event(False, key, mod)
    env.draw_less = False
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    #state = pi.get_initial_state()
    while 1:
        human_wants_restart = False
        saver.restore(sess, 'models/%s' % experiment)
        sn = env.reset()
        frame = 0
        r = 0
        uscore = 0
        state = {}
        ts1 = time.time()
        while 1:
            s = sn
            #a = agent.control(s, rng)
            stochastic = 1
            a, vpred, *state = pi.act(stochastic, s, *state)
            print (a)
            r = 0
            sn, rplus, done, info = env.step(a)
            r += rplus
            #if frame > env.spec.timestep_limit:
            #    done = True
            uscore += r
            frame += 1
            env.render("human")
            env.viewer.window.set_caption("%09.2f in %05i" % (uscore, frame))
            if done:
                for _ in range(10):
                    time.sleep(0.1)
                    env.viewer.window.dispatch_events()
            while human_sets_pause:
                time.sleep(0.1)
                env.viewer.window.dispatch_events()
            if done or human_wants_restart: break
            #if "print_state" in type(env).__dict__:
            #    env.print_state(sn)
        ts2 = time.time()
        print("score=%0.2f length=%i fps=%0.2f" % (uscore, frame, frame/(ts2-ts1)))
        env.monitor.close()

