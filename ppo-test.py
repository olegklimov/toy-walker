import os, sys, subprocess
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tinkerbell.logger as logger
from rl_algs.common import set_global_seeds
from rl_algs.common.mpi_fork import mpi_fork
from rl_algs import pposgd
from rl_algs.pposgd.mlp_policy import MlpPolicy

from mpi4py import MPI
num_cpu = 8

import gym
experiment = sys.argv[1]
print("experiment_name: '%s'" % experiment)
#env_id = "LunarLanderContinuous-v2"
env_id = "BipedalWalker-v2"
max_timesteps = 2000000
seed = 1339

demo = len(sys.argv)>2 and sys.argv[2]=="demo"


# ------------------------------- network ----------------------------------

policy_kwargs = dict(
    hid_size=120,
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
                pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
            self.pd = pdtype.pdfromflat(pdparam)

            # ---
            self.state_in = []
            self.state_out = []

            stochastic = tf.placeholder(dtype=tf.bool, shape=())
            ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
            self._act = U.function([stochastic, ob], [ac, self.vpred])

        def act(self, stochastic, ob):
            ac1, vpred1 =  self._act(stochastic, ob[None])
            return ac1[0], vpred1[0]
        def get_variables(self):
            return tf.get_collection(tf.GraphKeys.VARIABLES, self.scope)
        def get_trainable_variables(self):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        def get_initial_state(self):
            return []

    return ModifiedPolicy(name=name, ob_space=ob_space, ac_space=ac_space, **policy_kwargs)


# ------------------------- learn -----------------------------

if not demo:
    learn_kwargs = dict(
        timesteps_per_batch=2048, # horizon
        max_kl=0.02, clip_param=0.2, entcoeff=0.01, # objective
        klcoeff=0.003, adapt_kl=0,
        optim_epochs=24, optim_stepsize=3e-4, optim_batchsize=64, linesearch=True, # optimization
        gamma=0.99, lam=0.95, # advantage estimation
        )

    # classic
    #optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=16, linesearch=True, # optimization
    #gamma=0.99, lam=0.9, # advantage estimation

    def train():
        whoami  = mpi_fork(num_cpu)
        print("MPI whoami == '%s'" % whoami, flush=True)
        if whoami == "parent":
            return
        import rl_algs.common.tf_util as U
        rank = MPI.COMM_WORLD.Get_rank()
        print("MPI Rank == '%i'" % rank, flush=True)
        #if rank != 0:
        #    logger.set_level(logger.DISABLED)

        logger.set_expt_dir("progress")
        if rank==0:
            tab_fn = "%s.csv" % (experiment)
            log_fn = "%s.log" % (experiment)
            with open(os.path.join("progress", log_fn), "w"): pass
            with open(os.path.join("progress", tab_fn), "w"): pass
            logger.add_tabular_output(tab_fn)
            logger.add_text_output(log_fn)
            logger.info(subprocess.Popen(['git', 'diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout.read())
        logger.remove_text_output(sys.stdout)

        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
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

        #with sess:
        set_global_seeds(seed)
        env = gym.make(env_id)
        #gym.logger.setLevel(logging.WARN)
        env.seed(seed + 10000*rank)
        env.monitor.start(os.path.join(logger.get_expt_dir(), "monitor"), force=True, video_callable=False)
        if rank==0: learn_kwargs["snapshot_callback"] = save_policy
        pposgd.learn(env, policy_fn, max_timesteps=max_timesteps, **learn_kwargs)
        env.monitor.close()

    train()

else: # demo
    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.InteractiveSession(config=config)

    env = gym.make(env_id)
    #env.monitor.start("demo", force=True)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    tf.global_variables_initializer().run()

    print("Loading '%s'" % experiment)
    tv_list = pi.get_trainable_variables()
    saver = tf.train.Saver(var_list=tv_list)

    #state = pi.get_initial_state()
    while 1:
        saver.restore(sess, 'models/%s' % experiment)
        sn = env.reset()
        ts = 0
        r = 0
        uscore = 0
        state = {}
        while 1:
            s = sn
            #a = agent.control(s, rng)
            stochastic = 0 #np.zeros( (1,1) )
            a, vpred, *state = pi.act(stochastic, s, *state)
            r = 0
            sn, rplus, done, info = env.step(a)
            r += rplus
            if ts > env.spec.timestep_limit:
                done = True
            uscore += r
            ts += 1
            if done: break
            env.render("human")
        print("score=%0.2f length=%i" % (uscore, ts))
        #env.monitor.close()

