import os, sys, subprocess, time
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
if rank!=0:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
from tqdm import tqdm

import tensorflow as tf

import tinkerbell.logger as logger
from rl_algs.common.mpi_fork import mpi_fork
from rl_algs import pposgd
#from rl_algs.pposgd.mlp_policy import MlpPolicy
#from rl_algs.sandbox.hoj.common import logx as logger
#import rl_algs.sandbox.oleg.evolution

num_cpu = 8

import gym
from gym.envs.registration import register
from gym.wrappers import SkipWrapper
skip_wrap = lambda x: x
#skip_wrap = SkipWrapper(2)

env_id = sys.argv[1]
experiment = sys.argv[2]
max_timesteps = 16000000 if env_id=='CommandWalker-v0' else 2000000

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


# --------------------------------- xp -------------------------------------

class ExperienceGenerator:
    def __init__(self, env, REPLAY_BUFFER_DEPTH, BATCH):
        self.env = env
        self.REPLAY_BUFFER_DEPTH = REPLAY_BUFFER_DEPTH
        self.BATCH = BATCH

    def reopen(self, mode):
        #self.ob    = np.zeros( [REPLAY_BUFFER_DEPTH] + list(env.observation_space.shape), dtype=np.float32 )
        self.ob    = np.memmap("ramdisk/%s_mmap_ob" % env_id, mode=mode, shape=[REPLAY_BUFFER_DEPTH] + list(env.observation_space.shape), dtype=np.float32)
        assert len(env.action_space.shape)==1
        #self.a     = np.zeros( [REPLAY_BUFFER_DEPTH,env.action_space.shape[0]], dtype=np.float32 )
        self.a     = np.memmap("ramdisk/%s_mmap_a" % env_id, mode=mode, shape=[REPLAY_BUFFER_DEPTH,env.action_space.shape[0]], dtype=np.float32)
        #self.obn   = np.zeros( [REPLAY_BUFFER_DEPTH] + list(env.observation_space.shape), dtype=np.float32 )
        self.obn   = np.memmap("ramdisk/%s_mmap_obn" % env_id, mode=mode, shape=[REPLAY_BUFFER_DEPTH] + list(env.observation_space.shape), dtype=np.float32)
        #self.r     = np.zeros( [REPLAY_BUFFER_DEPTH,1], dtype=np.float32 )
        self.r     = np.memmap("ramdisk/%s_mmap_r" % env_id, mode=mode, shape=[REPLAY_BUFFER_DEPTH,1], dtype=np.float32)
        self.rsign = np.zeros( [REPLAY_BUFFER_DEPTH,2], dtype=np.float32 )
        self.total_reward = 0.0
        self.total_episodes = 0
        self.cursor = 2**31
        self.epoch = 0

    def gather_experience(self, policy, force_rerun=False):
        try:
            xp.export_viz_open(dir_jpeg, "r+")
        except:
            force_rerun = True
            xp.export_viz_open(dir_jpeg, "w+")
        if force_rerun:
            self.run_a_lot_of_rollouts(policy)
        for i in range(self.REPLAY_BUFFER_DEPTH):
            self.rsign[i,0] = 1 if self.r[i,0]<0 else 0
            self.rsign[i,1] = 0 if self.r[i,0]<0 else 1

    def run_a_lot_of_rollouts(self, policy):
        done = True
        for i in tqdm(range(self.REPLAY_BUFFER_DEPTH)):
            if done:
                ob = self.env.reset()
                done = False
                self.total_episodes += 1
            a, _  = policy.act(1, ob)
            self.a[i] = a
            self.ob[i] = ob
            ob, r, done, _ = self.env.step(a)
            self.obn[i] = ob
            self.r[i] = r
            self.total_reward += r

    def next_batch(self):
        if self.cursor+self.BATCH > self.REPLAY_BUFFER_DEPTH:
            sh = np.random.choice(self.REPLAY_BUFFER_DEPTH, size=self.REPLAY_BUFFER_DEPTH, replace=False)   # sh is list of indexes
            self.shuffled_ob    = self.ob[sh]
            self.shuffled_a     = self.a[sh]
            self.shuffled_obn   = self.obn[sh]
            self.shuffled_r     = self.r[sh]
            self.shuffled_rsign = self.rsign[sh]
            self.cursor = 0
            self.epoch += 1
        ret = (
            self.shuffled_ob[   self.cursor : self.cursor+self.BATCH],
            self.shuffled_a[    self.cursor : self.cursor+self.BATCH],
            self.shuffled_obn[  self.cursor : self.cursor+self.BATCH],
            self.shuffled_r[    self.cursor : self.cursor+self.BATCH],
            self.shuffled_rsign[self.cursor : self.cursor+self.BATCH])
        self.cursor += self.BATCH
        return ret


# ------------------------------- network ----------------------------------

policy_kwargs = dict(
    hid_size=196,
    num_hid_layers=2
    )

def policy_fn(name, env, data_init=False):
    import rl_algs.common.tf_util as U
    import rl_algs.common.tf_weightnorm as W
    from rl_algs.common.distributions import make_pdtype
    from rl_algs.common.mpi_running_mean_std import RunningMeanStd
    class ModifiedPolicy(object):
        recurrent = False

        def __init__(self, name, *args, **kwargs):
            with tf.variable_scope(name):
                self._init(*args, **kwargs)
                self.trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            print("Policy summary:")
            for v in self.trainable:
                print("\t'%s'" % v.name)

        def _init(self, ob_space, ac_space, hid_size, num_hid_layers, oldschool):
            assert isinstance(ob_space, gym.spaces.Box)

            self.pdtype = pdtype = make_pdtype(ac_space)
            sequence_length = None

            ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
            self.ob = ob

            if True:
                with tf.variable_scope("retfilter"):
                    self.ret_rms = RunningMeanStd()
            else:
                class Dummy:
                    def update(self, x):
                        pass
                self.ret_rms = Dummy()
                self.ret_rms.mean = 0.0
                self.ret_rms.std = 250.0

            if oldschool:
                # value est
                x = ob
                for i in range(num_hid_layers):
                    x = tf.nn.relu(U.dense(x, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
                self.vpredz = U.dense(x, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean

                # action
                x = ob
                for i in range(num_hid_layers):
                    x = tf.nn.relu(U.dense(x, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
                self.preaction = x

            else:
                self.wn_init = W.WeightNormInitializer()
                x = ob
                skip1 = x
                x = tf.nn.relu( W.dense_wn(x, 128, "crazy1", wn_init=self.wn_init) )
                skip2 = x
                x = tf.nn.relu( W.dense_wn(x,  96, "crazy2", wn_init=self.wn_init) )
                skip3 = x
                x = tf.nn.relu( W.dense_wn(x,  96, "crazy3", wn_init=self.wn_init) )
                skip4 = x
                x = tf.nn.relu( W.dense_wn(x,  64, "crazy4", wn_init=self.wn_init) )
                skip5 = x
                #self.preaction = U.concatenate( [skip1,skip2,skip3,skip4,skip5,skip6,skip7], axis=1 )
                self.preaction = x
                print("preaction shape", self.preaction.get_shape().as_list())
                self.vpredz = W.dense_wn(self.preaction, 1, "crazy_v")[:,0]
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean

                #self.aux_rewardclass_gt = U.get_placeholder(name="aux_rewardclass_gt", dtype=tf.float32, shape=[sequence_length] + [2])
                #aux_r_tensor = W.dense_wn(x, 2, "aux_reward_sign_logits", wn_init=self.wn_init)
                #self.aux_rewardclass_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(aux_r_tensor, self.aux_rewardclass_gt))
                #self.aux_rewardclass_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(aux_r_tensor, 1), tf.argmax(self.aux_rewardclass_gt, 1)), tf.float32))
                #tf.summary.scalar('aux_rewardclass_loss', self.aux_rewardclass_loss)
                #tf.summary.scalar('aux_rewardclass_accuracy', 100*self.aux_rewardclass_accuracy)

            # action pd
            if isinstance(ac_space, gym.spaces.Box):
                mean = U.dense(self.preaction, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer)
                #pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
                pdparam = U.concatenate([ 2.0*tf.nn.tanh(mean), mean * 0.0 - 0.2 + 0.0*logstd ], axis=1)
                # -0.2 => 0.8 (works for ppo/walking)
                # -0.5 => 0.6
                # -1.0        best for Lander
                # -1.6 => 0.2 (works for evolution)
                # -2.3 => 0.1
            else:
                bug()
                pdparam = U.dense(self.preaction, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
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
            return self.trainable
        def get_trainable_variables(self):
            return self.trainable
        def get_initial_state(self):
            return []
        def load(self, fn):
            print("LOAD '%s'" % fn)
            saver = tf.train.Saver(var_list=self.trainable)
            saver.restore(tf.get_default_session(), 'models/%s' % fn)
            sys.stdout.flush()
        def save(self, fn):
            print("SAVE\ntrainable:")
            for v in self.trainable:
                print("\t'%s'" % v.name)
            saver = tf.train.Saver(var_list=self.trainable)
            saver.save(tf.get_default_session(), 'models/%s' % fn)

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = ModifiedPolicy(name=name, ob_space=ob_space, ac_space=ac_space, oldschool=False, **policy_kwargs)
    U.initialize()

    def data_init_func(train_writer, test_writer):
        print("Data init of %s" % name)
        other_env = gym.make(env_id)
        other_pi = ModifiedPolicy(name="pi", ob_space=ob_space, ac_space=ac_space, oldschool=True, **policy_kwargs)
        U.initialize()
        other_pi.load("aux01_196_3")

        SAMPLES_DATA_INIT = 640*1000
        BATCH = 128
        xp = ExperienceGenerator(other_env, SAMPLES_DATA_INIT, BATCH)
        xp.gather_experience(other_pi)
        print("Mean reward of %i episodes: %0.1f" % (xp.total_episodes, xp.total_reward / xp.total_episodes))

        ob, a, obn, r, rsign = xp.next_batch()
        pi.wn_init.data_based_initialization({ pi.ob: ob })
        pi.wn_init.dump_to_tf_summary()
        #tf.summary.image("observation_batch", pi.ob.reshape([1,SAMPLES_DATA_INIT] + list(ob_space.shape) + [1]))

        print("Supervised reward train:")
        with tf.variable_scope("reward_supervised_train"):
            placeholder_reward_gt = tf.placeholder(dtype=tf.float32, shape=[BATCH,2], name="reward_gt")
            placeholder_xp_a = tf.placeholder(dtype=tf.float32, shape=[BATCH,env.action_space.shape[0]], name="xp_a")
            preaction_concat_xp_action = U.concatenate([placeholder_xp_a, pi.preaction], axis=1)
            x = tf.nn.relu( U.dense(preaction_concat_xp_action, 32, "crazy_reward_nonlin") )
            reward_approx = U.dense(x, 2, "crazy_reward_2way")
            reward_gt = tf.placeholder(dtype=tf.float32, shape=[BATCH,2])
            #batch_of_losses = tf.reduce_sum(tf.square(placeholder_reward_gt-reward_approx), axis=[1])
            #print("batch_of_losses", batch_of_losses.get_shape().as_list())
            #reward_loss = tf.reduce_mean(batch_of_losses)
            reward_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(reward_approx, placeholder_reward_gt))
            reward_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
            for v in pi.trainable + reward_trainable:
                print("\tREW '%s'" % v.name)
            reward_adam = tf.train.AdamOptimizer(0.0005, beta1=0.5).minimize(reward_loss, var_list=pi.trainable + reward_trainable)
            #reward_adam = tf.train.GradientDescentOptimizer(0.0005).minimize(reward_loss, var_list=pi.trainable + reward_trainable)
        tf.summary.scalar('reward_loss', reward_loss)
        #tf.summary.image("reward_diff", (ob).reshape([1,SAMPLES_DATA_INIT] + list(ob_space.shape) + [1]))
        U.initialize()
        merged = tf.summary.merge_all()

        print("Unlearned:")
        dump_me = tf.get_default_session().run(reward_approx, feed_dict={ pi.ob: ob, placeholder_xp_a: a })
        for i in range(20):
            print("step %i real %0.3f predicted [%0.3f,%0.3f]" % (i, r[i], dump_me[i,0], dump_me[i,1]), flush=True)
        for i in tqdm(range(50000)):
            ob, a, obn, r, rsign = xp.next_batch()
            summary,_,loss = tf.get_default_session().run( [merged,reward_adam,reward_loss], feed_dict={
                placeholder_xp_a: a,
                pi.ob: ob,
                placeholder_reward_gt: rsign
                } )
            train_writer.add_summary(summary, i)
        print("Learned in %i epochs:", xp.epoch)
        dump_me = tf.get_default_session().run(reward_approx, feed_dict={ pi.ob: ob, placeholder_xp_a: a })
        for i in range(20):
            print("step %i real %0.3f predicted [%0.3f,%0.3f]" % (i, r[i], dump_me[i,0], dump_me[i,1]), flush=True)

    if data_init and rank==0:
        LOG_DIR = "ramdisk/"
        import os, shutil
        LOG_TEST_DIR  = LOG_DIR + "/test_%s" % experiment
        LOG_TRAIN_DIR = LOG_DIR + "/train_%s" % experiment
        shutil.rmtree(LOG_TEST_DIR,  ignore_errors=True)
        shutil.rmtree(LOG_TRAIN_DIR, ignore_errors=True)
        os.makedirs(LOG_TEST_DIR)
        os.makedirs(LOG_TRAIN_DIR)
        train_writer = tf.summary.FileWriter(LOG_TRAIN_DIR, tf.get_default_session().graph)
        test_writer  = tf.summary.FileWriter(LOG_TEST_DIR)
        data_init_func(train_writer, test_writer)
        train_writer.close()
        test_writer.close()

    return pi


# ------------------------- learn -----------------------------

if not demo and not manual:
    def train():
        learn_kwargs = dict(
            timesteps_per_batch=1024, # horizon
            max_kl=0.03, clip_param=0.2, entcoeff=0.00, # objective
            #klcoeff=0.01, adapt_kl=0,
            klcoeff=0.001, adapt_kl=0.03,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64, linesearch=True, # optimization
            gamma=0.99, lam=0.95, # advantage estimation (try lambda .99)
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
            pi.save(experiment)

        def load_policy(pi):
            if not load_previous_experiment: return
            pi.load(load_previous_experiment)

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
    pi = policy_fn("newpi", env)

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
        pi.load(experiment)
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

