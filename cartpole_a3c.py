import numpy as np
import threading
import multiprocessing
import tensorflow as tf
from brain_a3c_discrete import A3CWorker

if __name__ == '__main__':
    LR_A = 0.006 #actor学习率
    LR_C = 0.006 #critic学习率
    maxepisode = 1000  #每个线程最大回合数
    maxsteps = 20000   #每个回合最大步数

    np.random.seed(1)
    sess = tf.Session()


    N_WORKERS = multiprocessing.cpu_count()  * 2  #线程数和CPU数有关
    #创建actor和critic的优化器
    opt_a = tf.train.RMSPropOptimizer(LR_A)
    opt_c = tf.train.RMSPropOptimizer(LR_C)

    #创建全局ACNET，仅用于更新参数
    globalacnet = A3CWorker('global',sess,None,opt_a,opt_c,maxepisode,maxsteps).acnet

    #创建worker
    workers=[]
    for i in range(N_WORKERS):
        scope = 'worker{}'.format(i)
        worker = A3CWorker('worker-%i' % i, sess,globalacnet,opt_a,opt_c,maxepisode,maxsteps)
        workers.append(worker)

    sess.run(tf.global_variables_initializer())
    COORD = tf.train.Coordinator()

    #启动线程
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=lambda: worker.work())
        worker_threads.append(t)
        t.start()
    COORD.join(worker_threads)
