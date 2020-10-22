import numpy as np
import tensorflow as tf
import gym
import threading

'''离散动作A3C'''
tf.set_random_seed(1)

#CartPole
ENV_NAME = 'CartPole-v0'

locker = threading.Lock()

#网络
class A3CNet(object):
    def __init__(self,scope, sess, opt_a,opt_c,n_state,n_action,globalac):
        self.scope = scope
        self.sess = sess
        self.n_state = n_state  #状态维度数 = 4
        self.n_action = n_action  #动作维度数 =2
        self.opt_a = opt_a
        self.opt_c = opt_c
        self.globalac = globalac

        #输入状态二维数组
        self.s = tf.placeholder(dtype='float32',shape=[None,self.n_state],name='s_input')
        #输入动作，一维数组
        self.a_his = tf.placeholder(dtype='int32',shape=[None,],name='a_his')
        #输入V值二维数组
        self.target_v = tf.placeholder(dtype='float32',shape=[None,1],name='target_v')
        self.buildnet()



    def buildnet(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('actor'):
                #actor a_prob = 200结点 ==>  n_action个结点， 激活softmax
                l1 = tf.layers.dense(inputs=self.s,units=200,activation=tf.nn.relu6)
                self.a_prob = tf.layers.dense(l1, self.n_action,activation=tf.nn.softmax)
            with tf.variable_scope('critic'):
                #critic 200结点 ==> 1个结点
                l2 = tf.layers.dense(inputs=self.s, units=200, activation=tf.nn.relu6)
                self.v = tf.layers.dense(l2, 1)

        #tderror 是现实v值和估计v的差
        tderror = self.target_v - self.v

        #critic损失是tderror的平方平均，最小二乖
        c_loss = tf.reduce_mean(tf.pow(tderror,2))

        #a_prob_log是动作估计值和现实值的交叉熵
        a_prob_log = -tf.reduce_sum(tf.log(self.a_prob * tf.one_hot(indices=self.a_his,depth=self.n_action)+1e-10),axis=1)
        #expv 动作交叉熵 * tderror，tderror的参数不要求偏导
        expv = a_prob_log * tf.stop_gradient(tderror)

        #动作概率熵
        entropy = tf.reduce_sum(-self.a_prob * tf.log(self.a_prob+1e-10),axis=1)
        #损失函数 动作交叉熵 + 系数 * 动作熵，求这个熵最小
        a_loss = tf.reduce_mean(expv + 0.01 * entropy)

        #提取theta参数
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/critic')

        if self.scope != 'global':
            #actor的梯度
            a_grad = tf.gradients(a_loss, self.a_params)
            #critic的梯度
            c_grad = tf.gradients(c_loss, self.c_params)

            #使用线程动作梯度更新全局动作theta参数
            self.update_a = self.opt_a.apply_gradients(zip(a_grad,self.globalac.a_params))
            #使用线程评价梯度更新全局评价theta参数
            self.update_c = self.opt_c.apply_gradients(zip(c_grad,self.globalac.c_params))

            #全局动作theta参数复制到线程动作参数
            self.pull_a = [l.assign(g) for (l,g) in zip(self.a_params,self.globalac.a_params)]
            #全局评价theta参数复制到线程评价参数
            self.pull_c = [l.assign(g) for (l,g) in zip(self.c_params,self.globalac.c_params)]

    #更新全局theta参数
    def updateglobal(self,s,a,v):
        self.sess.run([self.update_c,self.update_a],feed_dict={self.s: s,self.a_his: a,self.target_v: v})

    #将全局theta参数复制到线程
    def pullglobal(self):
        self.sess.run([self.pull_a,self.pull_c])

    #计算一个状态的回报V
    def calcV(self,s):
        v_value = self.sess.run(self.v,feed_dict={self.s: s})[0]
        return v_value[0]

    #由一个状态对应所有动作概率，返回概率最大的动作
    def choiceaction(self,s):
        # 返回所有动作的概率
        prob = self.sess.run(self.a_prob, feed_dict={self.s: s[None, :]})[0]
        action = np.random.choice(np.arange(self.n_action), p=prob.ravel())
        return action


#线程worker
class A3CWorker(object):
    def __init__(self,scope,sess,globalacnet,opt_a,opt_c,maxepisode,maxsteps):
        self.env = gym.make(ENV_NAME).unwrapped
        self.env.seed(1)
        #状态维度4
        self.n_state = self.env.observation_space.shape[0]
        #动作维度2
        self.n_action = self.env.action_space.n
        self.scope = scope
        #创建acnet
        self.acnet = A3CNet(scope, sess,  opt_a, opt_c, self.n_state,self.n_action,globalacnet)
        #最大回合
        self.maxepisode = maxepisode
        #一个回合最大步数
        self.maxsteps = maxsteps
        #每多个步更新一次全局参数
        self.STEP_ITER = 20
        #回报折现率
        self.gamma = 0.9
        #更用acnet选动作的概率
        self.epsilon = 0.10
        #更用acnet选动作的概率增加值
        self.epsilondelta = 0.007
        #更用acnet选动作的最大值，大于后为100%
        self.epsilonmax = 0.95

    def work(self):
        #创建s,a,r缓存，避免反复创建释放内存
        buffer_s = np.zeros((self.maxsteps,self.n_state),dtype='float32')
        buffer_a = np.zeros((self.maxsteps,),dtype='int32')
        buffer_r = np.zeros((self.maxsteps,1),dtype='float32')

        RENDER = False
        for epi in range(self.maxepisode):
            s = self.env.reset()
            runstep = 0
            #回合开始从全局获得参数
            self.acnet.pullglobal()
            memoryindex = 0
            for step in range(self.maxsteps):
                if RENDER:
                    self.env.render()

                #由状态s,策略epsilon greedy获得一个动作a
                a = self.choiceaction(s)
                s_, r, done, _ = self.env.step(a)
                runstep += 1

                #如果完成回报为-5
                if done: r = -5

                #保存到buffer中
                buffer_s[memoryindex,:] = s
                buffer_a[memoryindex] = a
                buffer_r[memoryindex,0] = r

                #动作达到步数或完成，更新全局theta
                if runstep%self.STEP_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        #如果完成，估计计算状态的回报v
                        v_s_ = self.acnet.calcV(s[None,:])

                    #将回报折现
                    for k in range(memoryindex - 1, -1, -1):
                        v_s_ = buffer_r[k,0] + self.gamma * v_s_
                        buffer_r[k,0] = v_s_

                    locker.acquire() #锁定lock更新global theta
                    #更新全局theta参数
                    self.acnet.updateglobal(buffer_s[:memoryindex,:],buffer_a[:memoryindex],buffer_r[:memoryindex,:])
                    locker.release()
                    #将更新后的全局参数更新回线程,回为有其它线程也在更新
                    self.acnet.pullglobal()
                    if done:
                        memoryindex = 0
                s = s_
                memoryindex += 1
                if memoryindex>=self.maxsteps:
                    memoryindex = 0

                if done:
                    break

            if runstep >= self.maxsteps and self.scope == 'worker-0':
                # 如果第0个worker达到最大步数，打开显示
                RENDER = True

            #增加espilon值，直到1
            self.epsilon += self.epsilondelta
            if self.epsilon >= self.epsilonmax:
                self.epsilon = 1.0
            #为了学习动作步数更多，取最大步数
            self.STEP_ITER = max(self.STEP_ITER, runstep)
            print('{}: epi={},run count={}'.format(self.scope, epi, runstep))


    def choiceaction(self,s):
        #由eplison概率来确定使用acnet选动作，还是随机选动作。
        if np.random.uniform() <= self.epsilon:
            action = self.acnet.choiceaction(s)
        else:
            action = np.random.choice(np.arange(self.n_action))
        return action
