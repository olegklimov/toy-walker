import os, sys, subprocess

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import tinkerbell.logger as logger
from rl_algs.common import set_global_seeds
from rl_algs.common.mpi_fork import mpi_fork
from rl_algs import pposgd
from rl_algs.pposgd.mlp_policy import MlpPolicy

from mpi4py import MPI
num_cpu = 4

import gym
experiment = sys.argv[1]
print("experiment_name: '%s'" % experiment)
env_id = "LunarLanderContinuous-v2"
max_timesteps = 2000000
seed = 1337

if experiment!="demo":
    policy_kwargs = dict(
        hid_size=16,
        num_hid_layers=2
        )

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, **policy_kwargs)

    learn_kwargs = dict(
        timesteps_per_batch=1024, # horizon
        max_kl=0.05, clip_param=0.2, entcoeff=0.01, # objective
        klcoeff=0.01, adapt_kl=0,
        optim_epochs=16, optim_stepsize=3e-4, optim_batchsize=64, linesearch=True, # optimization
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
            logger.info(subprocess.Popen('git diff', stdout=subprocess.PIPE).read())
        logger.remove_text_output(sys.stdout)

        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        sess = tf.InteractiveSession(config=config)

        def save_policy(pi):
            tv_list = pi.get_trainable_variables()
            print("SAVE")
            print("tainable:")
            for v in tv_list:
                print("\t'%s'" % v.name)
            saver = tf.train.Saver(var_list=tv_list)
            saver.save(sess, 'models/my-model')  #, global_step=0)

        #with sess:
        set_global_seeds(seed)
        env = gym.make(env_id)
        #gym.logger.setLevel(logging.WARN)
        env.seed(seed + 10000*rank)
        env.monitor.start(os.path.join(logger.get_expt_dir(), "monitor"), force=True, video_callable=False)
        pposgd.learn(env, policy_fn, max_timesteps=max_timesteps, snapshot_callback=save_policy, **learn_kwargs)
        env.monitor.close()

    train()

else:
    ac, vpred, *state = pi.act(stochastic, ob, *state)
