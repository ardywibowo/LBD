

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer


class MultiSIVAE_WImp_NewDropOut(object):
    
    def __init__(self, p_dims, q_dims=None, noise_dims=None,noise_prob=0.25,inject_in_between=0,num_lay_per_noise=1,test_mean_z=1,lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.num_lay_per_noise = num_lay_per_noise
        if noise_dims is None:
            self.noise_dims = q_dims[1:]
        else:
            self.noise_dims = noise_dims

        self.q_dims_final_in = [self.q_dims[0]]+[item for item in self.q_dims[1:] for i in range(self.num_lay_per_noise)]
        for dum_i in range(len(self.noise_dims)):
            self.q_dims_final_in[dum_i*self.num_lay_per_noise]=self.q_dims_final_in[dum_i*self.num_lay_per_noise]+self.noise_dims[dum_i]+ self.q_dims[0]*(dum_i!=0)
        self.q_dims_final_out=[self.q_dims[0]]+[item for item in self.q_dims[1:] for i in range(self.num_lay_per_noise)]
        
        self.noise_prob = noise_prob
        
        self.dims = self.q_dims + self.p_dims[1:]
        
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed
        
        self.test_mean_z=test_mean_z
        
        self.inject_in_between = inject_in_between
        
        self.input_noise_gen = tf.distributions.Bernoulli(noise_prob)
        self.layer_noise_gen = tf.distributions.Bernoulli(noise_prob)

        self.construct_placeholders()

    def construct_placeholders(self):
        
        self.K = tf.placeholder(tf.int32, shape=())
        
        self.K_u = tf.placeholder(tf.int32, shape=())
        self.K_ARM = tf.placeholder_with_default(1, shape=())
        
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        #self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)
        
    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits,axis=-1)
        
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        
        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        
        if self.test_mean_z==1:
            neg_ll_all=tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda: tf.reduce_sum(log_softmax_var * self.input_ph,axis=-1),lambda: tf.reduce_sum(log_softmax_var * tf.expand_dims(self.input_ph,axis=1),axis=-1))
        else:
            neg_ll_all=tf.reduce_sum(log_softmax_var * tf.expand_dims(self.input_ph,axis=1),axis=-1)
        
        
        
        neg_ll_scalar=-tf.reduce_mean(neg_ll_all)
        KL_scalar = tf.reduce_mean(KL)
        neg_ELBO_scalar = neg_ll_scalar + self.anneal_ph * KL_scalar + 2 * reg_var

        
        
        if self.test_mean_z==1:
            loss_iw = tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda: neg_ELBO_scalar, lambda: tf.reduce_mean(tf.reduce_logsumexp(-neg_ll_all + self.anneal_ph * KL,1)+tf.log(tf.cast(self.K,tf.float32)) ))#Only change this line for without importance sampling
            
        else:
            loss_iw = tf.reduce_mean(tf.reduce_logsumexp(-neg_ll_all + self.anneal_ph * KL,1)+tf.log(tf.cast(self.K,tf.float32)) )
            log_softmax_var = tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda:tf.reduce_logsumexp(log_softmax_var,axis=1),lambda:log_softmax_var)


        
        neg_ELBO = loss_iw + 2 * reg_var


        
        trainer1 = tf.train.AdamOptimizer(self.lr)
        gradvars_1=trainer1.compute_gradients(neg_ELBO,var_list=[self.weights_p,self.biases_p,self.weights_q,self.biases_q])
        train_op1 = trainer1.apply_gradients(gradvars_1)
        


        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll_scalar)
        tf.summary.scalar('KL', KL_scalar)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO_scalar)
        merged = tf.summary.merge_all()
        
        trainer2 = tf.train.AdamOptimizer(self.lr)
        gradvars_2=self.forward_pass_ARM()
        train_op2 = trainer2.apply_gradients([(gradvars_2,self.keep_prob_ph)])
        
        with tf.control_dependencies([train_op1,train_op2]):
            train_op = tf.no_op()

        return saver, log_softmax_var, neg_ELBO, train_op, merged
    
    def q_graph(self):

        mu_q, logvar_q = None, None
        
        eps2 = 1e-15
        
        xx_o = tf.nn.l2_normalize(self.input_ph, 1)
        
        
        xx_o=tf.expand_dims(xx_o,axis=1)
        xx_o = tf.tile(xx_o,[1,self.K,1])
        
        pp=tf.expand_dims(tf.expand_dims(tf.sigmoid(self.keep_prob_ph),axis=0),axis=0)
        
        drop_gen=tf.distributions.Bernoulli(logits=self.keep_prob_ph)
        xx = xx_o*tf.cast(drop_gen.sample([tf.shape(xx_o)[0],self.K]),tf.float32)/(pp+eps2)
        
        noise_counter=0
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            if i%self.num_lay_per_noise==0:
              if i==0:
                #cc = tf.cast(self.input_noise_gen.sample([tf.shape(xx_o)[0],self.K,self.noise_dims[noise_counter]]),tf.float32)
                #cc = cc/tf.sqrt(tf.cast(self.input_noise_gen.mean(),tf.float32)*self.noise_dims[noise_counter])
                h=xx#tf.concat([xx,cc],axis=2)
              else:
                  if self.inject_in_between==1:
                      cc = tf.minimum(tf.maximum(tf.cast(self.layer_noise_gen.sample([tf.shape(xx_o)[0],self.K,self.noise_dims[noise_counter]]),tf.float32),-1.0),1.0)
                      cc = cc/tf.sqrt(tf.cast(self.layer_noise_gen.mean(),tf.float32)*self.noise_dims[noise_counter])
                      xx = xx_o*tf.cast(drop_gen.sample([tf.shape(xx_o)[0],self.K]),tf.float32)/(pp+eps2)
                      h=tf.concat([h,xx,cc],axis=2)
              noise_counter = noise_counter+1

            h=tf.tensordot(h,w,axes=[[2],[0]])+ b
            
            if i != len(self.weights_q) - 1:
                h = tf.nn.relu(h)
            else:
                mu_q = h[:, :, :self.q_dims[-1]]
                logvar_q = h[:, :, self.q_dims[-1]:]


        return mu_q, logvar_q


    
    def p_graph_tile(self, z, do_tile):
        h = z

        if do_tile==1:
            for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):

                h=tf.tensordot(h,w,axes=[[2],[0]]) + b

                if i != len(self.weights_p) - 1:
                    h = tf.nn.relu(h)
        else:
            for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
                 h=tf.matmul(h,w)+b
 
             
                 if i != len(self.weights_p) - 1:
                     h = tf.nn.relu(h)
        return h

    def forward_pass(self):
        
        eps = 1e-11
        
        mu_q, logvar_q= self.q_graph()
        std_q = tf.exp(0.5 * logvar_q)
        epsilon = tf.random_normal(tf.shape(std_q))
        
        
        
        sampled_z = mu_q + epsilon * std_q

        
        log_prior_iw = -0.5*tf.reduce_sum(tf.square(sampled_z),2)
        sampled_z_dim=tf.expand_dims(sampled_z,axis=2)
        mu_q_dim=tf.expand_dims(mu_q,axis=1)
        std_q_dim=tf.expand_dims(std_q,axis=1)
        logvar_q_dim=tf.expand_dims(logvar_q,axis=1)
        ker=-0.5*tf.reduce_sum(tf.square(sampled_z_dim-mu_q_dim)/tf.square(std_q_dim+eps),3) - 0.5*tf.reduce_sum(logvar_q_dim,3)
        log_H_iw = tf.reduce_logsumexp(ker,2)-tf.log(tf.cast(self.K,tf.float32))
        
        KL=log_H_iw-log_prior_iw
        

        if self.test_mean_z==1:
            sampled_z = tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda:tf.reduce_mean(sampled_z,axis=1),lambda: sampled_z)
            logits = tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda:self.p_graph_tile(sampled_z,0),lambda: self.p_graph_tile(sampled_z,1))
        else:
            logits = self.p_graph_tile(sampled_z,1)
        

        
        return tf.train.Saver(), logits, KL
    
    def forward_pass_ARM(self):
        
        mu_q1, logvar_q1 = None, None
        mu_q2, logvar_q2 = None, None
        
        eps2 = 1e-15
        
        xx_o = tf.nn.l2_normalize(self.input_ph, 1)
        
        
        xx_o=tf.expand_dims(xx_o,axis=1)
        xx_o = tf.tile(xx_o,[1,self.K_u,1])
        
        pp=tf.expand_dims(tf.expand_dims(tf.sigmoid(self.keep_prob_ph),axis=0),axis=0)
        pp=tf.tile(pp,[tf.shape(xx_o)[0],self.K_u,1])
        pp_ = 1.0-pp
        
        
        drop_gen=tf.distributions.Uniform()
        
        uu=drop_gen.sample([tf.shape(xx_o)[0],1,self.p_dims[-1]])
        uu_all = tf.squeeze(uu)
        uu=tf.tile(uu,[1,self.K_u,1])
        xx1 = xx_o*tf.cast(uu>pp_,tf.float32)/(pp+eps2)
        xx2 = xx_o*tf.cast(uu<pp,tf.float32)/(pp+eps2)
        
        
        
        noise_counter=0
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            if i%self.num_lay_per_noise==0:
              if i==0:
                #cc = tf.cast(self.input_noise_gen.sample([tf.shape(xx_o)[0],self.K_u,self.noise_dims[noise_counter]]),tf.float32)
                #cc = cc/tf.sqrt(tf.cast(self.input_noise_gen.mean(),tf.float32)*self.noise_dims[noise_counter])
                ##h=tf.concat([xx,cc],axis=2)
                h1=xx1#tf.concat([xx1,cc],axis=2)
                h2=xx2#tf.concat([xx2,cc],axis=2)
              else:
                  if self.inject_in_between==1:
                      cc = tf.minimum(tf.maximum(tf.cast(self.layer_noise_gen.sample([tf.shape(xx_o)[0],self.K_u,self.noise_dims[noise_counter]]),tf.float32),-1.0),1.0)
                      cc = cc/tf.sqrt(tf.cast(self.layer_noise_gen.mean(),tf.float32)*self.noise_dims[noise_counter])
                      
                      uu=drop_gen.sample([tf.shape(xx_o)[0],1,self.p_dims[-1]])
                      uu_all = uu_all + tf.squeeze(uu)
                      uu=tf.tile(uu,[1,self.K_u,1])
                      xx1 = xx_o*tf.cast(uu>pp_,tf.float32)/(pp+eps2)
                      xx2 = xx_o*tf.cast(uu<pp,tf.float32)/(pp+eps2)
                      h1=tf.concat([h1,xx1,cc],axis=2)
                      h2=tf.concat([h2,xx2,cc],axis=2)
              noise_counter = noise_counter+1

            
            h1=tf.tensordot(h1,w,axes=[[2],[0]])+ b
            h2=tf.tensordot(h2,w,axes=[[2],[0]])+ b
            
            if i != len(self.weights_q) - 1:
                h1 = tf.nn.relu(h1)
                h2 = tf.nn.relu(h2)
            else:
                mu_q1 = h1[:, :, :self.q_dims[-1]]
                logvar_q1 = h1[:, :, self.q_dims[-1]:]
                mu_q2 = h2[:, :, :self.q_dims[-1]]
                logvar_q2 = h2[:, :, self.q_dims[-1]:]


        eps = 1e-11

        std_q1 = tf.exp(0.5 * logvar_q1)
        epsilon1 = tf.random_normal(tf.shape(std_q1))
        std_q2 = tf.exp(0.5 * logvar_q2)
        epsilon2 = tf.random_normal(tf.shape(std_q2))
        
        
        
        sampled_z1 = mu_q1 + epsilon1 * std_q1
        sampled_z2 = mu_q2 + epsilon2 * std_q2

        
        log_prior_iw1 = -0.5*tf.reduce_sum(tf.square(sampled_z1),2)
        sampled_z_dim1=tf.expand_dims(sampled_z1,axis=2)
        mu_q_dim1=tf.expand_dims(mu_q1,axis=1)
        std_q_dim1=tf.expand_dims(std_q1,axis=1)
        logvar_q_dim1=tf.expand_dims(logvar_q1,axis=1)
        ker1=-0.5*tf.reduce_sum(tf.square(sampled_z_dim1-mu_q_dim1)/tf.square(std_q_dim1+eps),3) - 0.5*tf.reduce_sum(logvar_q_dim1,3)
        log_H_iw1 = tf.reduce_logsumexp(ker1,2)-tf.log(tf.cast(self.K_u,tf.float32))
        
        KL1=log_H_iw1-log_prior_iw1
        
        log_prior_iw2 = -0.5*tf.reduce_sum(tf.square(sampled_z2),2)
        sampled_z_dim2=tf.expand_dims(sampled_z2,axis=2)
        mu_q_dim2=tf.expand_dims(mu_q2,axis=1)
        std_q_dim2=tf.expand_dims(std_q2,axis=1)
        logvar_q_dim2=tf.expand_dims(logvar_q2,axis=1)
        ker2=-0.5*tf.reduce_sum(tf.square(sampled_z_dim2-mu_q_dim2)/tf.square(std_q_dim2+eps),3) - 0.5*tf.reduce_sum(logvar_q_dim2,3)
        log_H_iw2 = tf.reduce_logsumexp(ker2,2)-tf.log(tf.cast(self.K_u,tf.float32))
        
        KL2=log_H_iw2-log_prior_iw2
        
        

        logits1 = self.p_graph_tile(sampled_z1,1)
            

        logits2 = self.p_graph_tile(sampled_z2,1)
        
        
        
        log_softmax_var1 = tf.nn.log_softmax(logits1,axis=-1)
        log_softmax_var2 = tf.nn.log_softmax(logits2,axis=-1)
        

        

        neg_ll_all1=tf.reduce_sum(log_softmax_var1 * tf.expand_dims(self.input_ph,axis=1),axis=-1)
        neg_ll_all2=tf.reduce_sum(log_softmax_var2 * tf.expand_dims(self.input_ph,axis=1),axis=-1)

        

        loss_iw1 = tf.reduce_mean(tf.reduce_logsumexp(-neg_ll_all1 + self.anneal_ph * KL1,1)+tf.log(tf.cast(self.K_u,tf.float32)) )
        loss_iw2 = tf.reduce_mean(tf.reduce_logsumexp(-neg_ll_all2 + self.anneal_ph * KL2,1)+tf.log(tf.cast(self.K_u,tf.float32)) )


        
        neg_ELBO1 = loss_iw1
        neg_ELBO2 = loss_iw2
        f_delta = neg_ELBO1 - neg_ELBO2
        return f_delta*tf.reduce_mean(uu_all-0.5,axis=0)
    def _construct_weights(self):
        
        self.keep_prob_ph = tf.get_variable(name="var_drop",shape=[self.p_dims[-1]],initializer=tf.initializers.random_normal(mean=0.0,stddev=0.1,seed=self.random_seed))

        self.weights_q, self.biases_q = [], []
        
        for i, (d_in, d_out) in enumerate(zip(self.q_dims_final_in[:-1], self.q_dims_final_out[1:])):
            if i == len(self.q_dims_final_in[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])
            
        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_p.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])
            
class MultiSIVAE_WOImp_NewDropOut(MultiSIVAE_WImp_NewDropOut):
    
    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits,axis=-1)
        
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        
        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        
        if self.test_mean_z==1:
            neg_ll_all=tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda: tf.reduce_sum(log_softmax_var * self.input_ph,axis=-1),lambda: tf.reduce_sum(log_softmax_var * tf.expand_dims(self.input_ph,axis=1),axis=-1))
        else:
            neg_ll_all=tf.reduce_sum(log_softmax_var * tf.expand_dims(self.input_ph,axis=1),axis=-1)
        
        
        
        neg_ll_scalar=-tf.reduce_mean(neg_ll_all)
        KL_scalar = tf.reduce_mean(KL)
        neg_ELBO_scalar = neg_ll_scalar + self.anneal_ph * KL_scalar + 2 * reg_var

        
        
        if self.test_mean_z==1:
            loss_iw = tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda: neg_ELBO_scalar, lambda: tf.reduce_mean(tf.reduce_mean(-neg_ll_all + self.anneal_ph * KL,1)) )#Only change this line for without importance sampling
            
        else:
            loss_iw = tf.reduce_mean(tf.reduce_mean(-neg_ll_all + self.anneal_ph * KL,1) )
            log_softmax_var = tf.cond(tf.logical_not(tf.equal(self.is_training_ph,1)),lambda:tf.reduce_logsumexp(log_softmax_var,axis=1),lambda:log_softmax_var)

      
        
        neg_ELBO = loss_iw + 2 * reg_var


        
        trainer1 = tf.train.AdamOptimizer(self.lr)
        gradvars_1=trainer1.compute_gradients(neg_ELBO,var_list=[self.weights_p,self.biases_p,self.weights_q,self.biases_q])
        train_op1 = trainer1.apply_gradients(gradvars_1)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll_scalar)
        tf.summary.scalar('KL', KL_scalar)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO_scalar)
        merged = tf.summary.merge_all()
        
        trainer2 = tf.train.AdamOptimizer(self.lr)
        gradvars_2=self.forward_pass_ARM()
        train_op2 = trainer2.apply_gradients([(gradvars_2,self.keep_prob_ph)])
        
        with tf.control_dependencies([train_op1,train_op2]):
            train_op = tf.no_op()

        return saver, log_softmax_var, neg_ELBO, train_op, merged
    
    def forward_pass_ARM(self):
        
        mu_q1, logvar_q1 = None, None
        mu_q2, logvar_q2 = None, None
        
        eps2 = 1e-15
        
        xx_o = tf.nn.l2_normalize(self.input_ph, 1)
        
        
        xx_o=tf.expand_dims(xx_o,axis=1)
        xx_o = tf.tile(xx_o,[1,self.K_u,1])
        
        pp=tf.expand_dims(tf.expand_dims(tf.sigmoid(self.keep_prob_ph),axis=0),axis=0)
        pp=tf.tile(pp,[tf.shape(xx_o)[0],self.K_u,1])
        pp_ = 1.0-pp
        
        
        drop_gen=tf.distributions.Uniform()
        
        uu=drop_gen.sample([tf.shape(xx_o)[0],1,self.p_dims[-1]])
        uu_all = tf.squeeze(uu)
        uu=tf.tile(uu,[1,self.K_u,1])
        xx1 = xx_o*tf.cast(uu>pp_,tf.float32)/(pp+eps2)
        xx2 = xx_o*tf.cast(uu<pp,tf.float32)/(pp+eps2)
        
        
        
        noise_counter=0
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            if i%self.num_lay_per_noise==0:
              if i==0:

                h1=xx1
                h2=xx2
              else:
                  if self.inject_in_between==1:
                      cc = tf.minimum(tf.maximum(tf.cast(self.layer_noise_gen.sample([tf.shape(xx_o)[0],self.K_u,self.noise_dims[noise_counter]]),tf.float32),-1.0),1.0)
                      cc = cc/tf.sqrt(tf.cast(self.layer_noise_gen.mean(),tf.float32)*self.noise_dims[noise_counter])
                      
                      uu=drop_gen.sample([tf.shape(xx_o)[0],1,self.p_dims[-1]])
                      uu_all = uu_all + tf.squeeze(uu)
                      uu=tf.tile(uu,[1,self.K_u,1])
                      xx1 = xx_o*tf.cast(uu>pp_,tf.float32)/(pp+eps2)
                      xx2 = xx_o*tf.cast(uu<pp,tf.float32)/(pp+eps2)
                      h1=tf.concat([h1,xx1,cc],axis=2)
                      h2=tf.concat([h2,xx2,cc],axis=2)
              noise_counter = noise_counter+1
 
            
            h1=tf.tensordot(h1,w,axes=[[2],[0]])+ b
            h2=tf.tensordot(h2,w,axes=[[2],[0]])+ b
            
            if i != len(self.weights_q) - 1:
                h1 = tf.nn.relu(h1)
                h2 = tf.nn.relu(h2)
            else:
                mu_q1 = h1[:, :, :self.q_dims[-1]]
                logvar_q1 = h1[:, :, self.q_dims[-1]:]
                mu_q2 = h2[:, :, :self.q_dims[-1]]
                logvar_q2 = h2[:, :, self.q_dims[-1]:]


        eps = 1e-11

        std_q1 = tf.exp(0.5 * logvar_q1)
        epsilon1 = tf.random_normal(tf.shape(std_q1))
        std_q2 = tf.exp(0.5 * logvar_q2)
        epsilon2 = tf.random_normal(tf.shape(std_q2))
        
        
        
        sampled_z1 = mu_q1 + epsilon1 * std_q1
        sampled_z2 = mu_q2 + epsilon2 * std_q2

        
        log_prior_iw1 = -0.5*tf.reduce_sum(tf.square(sampled_z1),2)
        sampled_z_dim1=tf.expand_dims(sampled_z1,axis=2)
        mu_q_dim1=tf.expand_dims(mu_q1,axis=1)
        std_q_dim1=tf.expand_dims(std_q1,axis=1)
        logvar_q_dim1=tf.expand_dims(logvar_q1,axis=1)
        ker1=-0.5*tf.reduce_sum(tf.square(sampled_z_dim1-mu_q_dim1)/tf.square(std_q_dim1+eps),3) - 0.5*tf.reduce_sum(logvar_q_dim1,3)
        log_H_iw1 = tf.reduce_logsumexp(ker1,2)-tf.log(tf.cast(self.K_u,tf.float32))
        
        KL1=log_H_iw1-log_prior_iw1
        
        log_prior_iw2 = -0.5*tf.reduce_sum(tf.square(sampled_z2),2)
        sampled_z_dim2=tf.expand_dims(sampled_z2,axis=2)
        mu_q_dim2=tf.expand_dims(mu_q2,axis=1)
        std_q_dim2=tf.expand_dims(std_q2,axis=1)
        logvar_q_dim2=tf.expand_dims(logvar_q2,axis=1)
        ker2=-0.5*tf.reduce_sum(tf.square(sampled_z_dim2-mu_q_dim2)/tf.square(std_q_dim2+eps),3) - 0.5*tf.reduce_sum(logvar_q_dim2,3)
        log_H_iw2 = tf.reduce_logsumexp(ker2,2)-tf.log(tf.cast(self.K_u,tf.float32))
        
        KL2=log_H_iw2-log_prior_iw2
        
        

        logits1 = self.p_graph_tile(sampled_z1,1)
            

        logits2 = self.p_graph_tile(sampled_z2,1)
        
        
        
        log_softmax_var1 = tf.nn.log_softmax(logits1,axis=-1)
        log_softmax_var2 = tf.nn.log_softmax(logits2,axis=-1)
        

        

        neg_ll_all1=tf.reduce_sum(log_softmax_var1 * tf.expand_dims(self.input_ph,axis=1),axis=-1)
        neg_ll_all2=tf.reduce_sum(log_softmax_var2 * tf.expand_dims(self.input_ph,axis=1),axis=-1)

        

        loss_iw1 = tf.reduce_mean(tf.reduce_mean(-neg_ll_all1 + self.anneal_ph * KL1,1) )
        loss_iw2 = tf.reduce_mean(tf.reduce_mean(-neg_ll_all2 + self.anneal_ph * KL2,1) )


        
        neg_ELBO1 = loss_iw1
        neg_ELBO2 = loss_iw2
        f_delta = neg_ELBO1 - neg_ELBO2
        return f_delta*tf.reduce_mean(uu_all-0.5,axis=0)


