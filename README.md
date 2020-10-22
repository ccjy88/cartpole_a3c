# cartpole_a3c
cartpole with A3C Asynchronous Advantage Actor-Critic


使用 Asynchronous Advantage Actor-Critic(A3C)算法，使用多线程，实现每个线程学习100多个回合，然后
一个回合20000步杆子不倒。

CartPole是http://gym.openai.com/envs/CartPole-v0/ 这个网站提供的一个杆子不倒的测试环境。
CartPole环境返回一个状态包括位置、加速度、杆子垂直夹角和角加速度。玩家控制左右两个方向使杆子不倒。
杆子倒了或超出水平位置限制就结束一个回合。一个回合中杆不倒动作步数越多越好。

基本算法是Policy Gradident Actor-Critic。A3C在多线程上，运行多个独立环境和独立智能体agent并行计算
达到缩短训练时间的效果。
只要硬件条件允许，线程数越多，学习越快。

每个状态的V值，损失函数和Actor-Critic一样的
        #tderror 是现实v值和估计v的差
        tderror = self.target_v - self.v

        #critic损失是tderror的平方平均，最小二乖
        c_loss = tf.reduce_mean(tf.pow(tderror,2))

动作的损失函数是最小化的动作和现实误差的交叉熵，加动作的熵。
        #a_prob_log是动作估计值和现实值的交叉熵
        a_prob_log = -tf.reduce_sum(tf.log(self.a_prob * tf.one_hot(indices=self.a_his,depth=self.n_action)+1e-10),axis=1)
        #expv 动作交叉熵 * tderror
        expv = a_prob_log * tf.stop_gradient(tderror)

        #动作概率熵
        entropy = tf.reduce_sum(-self.a_prob * tf.log(self.a_prob+1e-10),axis=1)
        #损失函数 动作交叉熵 + 系数 * 动作熵，求这个熵最小
        a_loss = tf.reduce_mean(expv + 0.01 * entropy)

A3C独特之处有一个全局的参数，每次进行更新theta时是更新全局的参数theta,
            #使用线程动作梯度更新全局动作theta参数
            self.update_a = self.opt_a.apply_gradients(zip(a_grad,self.globalac.a_params))
            #使用线程评价梯度更新全局评价theta参数
            self.update_c = self.opt_c.apply_gradients(zip(c_grad,self.globalac.c_params))
	多线程更新参数时增加了线程锁
    locker.acquire() #锁定lock更新global theta
    #更新全局theta参数
    self.acnet.updateglobal(buffer_s[:memoryindex,:],buffer_a[:memoryindex],buffer_r[:memoryindex,:])
    locker.release()


从全局参数复制到线程参数
            #全局动作theta参数复制到线程动作参数
            self.pull_a = [l.assign(g) for (l,g) in zip(self.a_params,self.globalac.a_params)]
            #全局评价theta参数复制到线程评价参数
            self.pull_c = [l.assign(g) for (l,g) in zip(self.c_params,self.globalac.c_params)]

因为多个线程多个独立环境独立智能体分别并行计算获得theta梯度，并行更新全局的theta参数，所以学习速度更快。
在硬件支持的情况下， 线程越多，学习越快。

多线程更新全局参数加了locker.acquire()和locker.release(),并不确定是不是必要。多线程更新一般需要加锁。





