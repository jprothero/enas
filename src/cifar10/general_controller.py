import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.controller import Controller
from src.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class GeneralController(Controller):
  def __init__(self,
               search_for="both",
               search_whole_channels=False,
               num_layers=4,
               num_branches=6,
               out_filters=48,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               temperature=None,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               entropy_weight=None,
               clip_mode=None,
               grad_bound=None,
               use_critic=False,
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               skip_target=0.8,
               skip_weight=0.5,
               name="controller",
               *args,
               **kwargs):

    print "-" * 80
    print "Building ConvController"

    self.search_for = search_for
    self.search_whole_channels = search_whole_channels
    self.num_layers = num_layers
    self.num_branches = num_branches
    self.out_filters = out_filters

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers 
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.entropy_weight = entropy_weight
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.use_critic = use_critic
    self.bl_dec = bl_dec

    self.skip_target = skip_target
    self.skip_weight = skip_weight

    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name

    class AlphaZero:
      def __init__(self):
        self.root_node = {
          "children": []
        }
        self.curr_node = self.root_node

      def uct_choice(self, curr_node): return curr_node["children"][curr_node["max_uct"]]

      def select(self, root_node):
          curr_node = root_node

          while curr_node.children is not None:
              curr_node = self.uct_choice(curr_node)

          return curr_node


      def U_func(self, P, N): return P/(1+N)


      def Q_func(self, W, N): return W/N


      def backup(self, expanded_node, value):
          node = expanded_node
          while node["parent"] is not None:
              node = self.update_node(node, value)
              node = node["parent"]

          return node


      def update_node(self, node, value):
          node["N"] += 1
          node["W"] += value
          node["Q"] = self.Q_func(W=node["W"], N=node["N"])
          node["U"] = self.U_func(P=node["P"], N=node["N"])
          UCT = node["Q"] + node["U"]
          if UCT > node["parent"]["max_uct"]["score"]:
              node["parent"]["max_uct"]["score"] = UCT
              node["parent"]["max_uct"]["idx"] = node["idx"]

          return node


      def expand(self, curr_node, policy, value):
          curr_node["children"] = []
          curr_node["max_uct"] = {}

          max_U = 0
          max_U_idx = None

          for i, p in enumerate(policy):
              U = self.U_func(p, 0)

              child = {
                  "N": 0,
                  "W": 0,
                  "Q": 0,
                  "U": U,
                  "P": p,
                  "parent": curr_node,
                  "idx": i
              }

              curr_node["children"].append(child)

              if U > max_U:
                  max_U = U
                  max_U_idx = i

          curr_node["max_uct"] = {
              "score": max_U, "idx": max_U_idx
          }

          return curr_node, value


      def choose_real_move(self, node):
          child_visits_probas = node["child_visits"]/node["child_visits"].sum()

          action = np.random.choice(
              node["child_visits"], p=child_visits_probas)

          return action, child_visits_probas

    #So I dont have time to do this right now
    #the idea is basically have a self.AZ which has the alpha zero methods
    #run build sampler NUM_SIMS times
    #whenever you hit a decision expand, backpropagate, and stay on the same node,
    #then use UCT to select the best node (alternative is we can do a random choice)
    #with the probas, but this is fine for now

    #in theory it should just work because it will follow the decision flow of the program
    #the num children will change dynamically based on the policy size, and on subsequent
    #runs of build sampler we will have stronger search probas, and use them 
    #to improve the base policy

    self.az = AlphaZero()


    self.root_node = {
      "children": None
      , "parent": None
      , "itrs": 0
      , "N": 0.0
    }

    self._create_params()

    NUM_SIMS=10
    for _ in range(NUM_SIMS):
      self._build_sampler()
    self._build_sampler(visits=True)

  def _create_params(self):
    initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    with tf.variable_scope(self.name, initializer=initializer):
      with tf.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in xrange(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable(
              "w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])
      if self.search_whole_channels:
        with tf.variable_scope("emb"):
          self.w_emb = tf.get_variable(
            "w", [self.num_branches, self.lstm_size])
        with tf.variable_scope("softmax"):
          self.w_soft = tf.get_variable(
            "w", [self.lstm_size, self.num_branches])
      else:
        self.w_emb = {"start": [], "count": []}
        with tf.variable_scope("emb"):
          for branch_id in xrange(self.num_branches):
            with tf.variable_scope("branch_{}".format(branch_id)):
              self.w_emb["start"].append(tf.get_variable(
                "w_start", [self.out_filters, self.lstm_size]));
              self.w_emb["count"].append(tf.get_variable(
                "w_count", [self.out_filters - 1, self.lstm_size]));

        self.w_soft = {"start": [], "count": []}
        with tf.variable_scope("softmax"):
          for branch_id in xrange(self.num_branches):
            with tf.variable_scope("branch_{}".format(branch_id)):
              self.w_soft["start"].append(tf.get_variable(
                "w_start", [self.lstm_size, self.out_filters]));
              self.w_soft["count"].append(tf.get_variable(
                "w_count", [self.lstm_size, self.out_filters - 1]));

      with tf.variable_scope("critic"):
        self.w_critic = tf.get_variable("value", [self.lstm_size, 1])

      with tf.variable_scope("attention"):
        self.w_attn_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
        self.w_attn_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
        self.v_attn = tf.get_variable("v", [self.lstm_size, 1])
  
  def single_select(self, curr_node, visits):
    if not visits:
      if curr_node["children"] is not None:

        ucts = [child["uct"] for child in curr_node["children"]]
        ucts = tf.stack(ucts)
        idx = tf.argmax(ucts)
        with tf.Session() as sess:
          curr_node = curr_node["children"][idx.eval()]
        # curr_node = tf.gather(curr_node["children"], tf_look_up[idx])
      else:
        idx = None
    else:
      idx = tf.argmax(curr_node["children"]["N"])
      curr_node = curr_node["children"][idx]
    
    return curr_node, idx

  def expand(self, curr_node, logit):
    curr_node["children"] = [] 
  
    # curr_node["policy"]
    policy = tf.nn.softmax(logit)

    policy_dim = policy.get_shape().as_list()[0]

    for i in enumerate(range(policy_dim)):
        child = {
            "N": 0,
            "W": 0,
            "Q": 0,
            "U": policy[i],
            "P": policy[i],
            "uct": policy[i],
            "children": None,
            "parent": curr_node,
        }

        curr_node["children"].extend([child])
    
    return curr_node

  # def expand(self, curr_node, logit):
  #   curr_node["children"] = []

  #   policy =  tf.nn.softmax(logit)

  #   for i, p in enumerate(policy):
  #       child = {
  #           "N": 0,
  #           "W": 0,
  #           "Q": 0,
  #           "U": p,
  #           "P": p,
  #           "uct": p,
  #           "children": None,
  #           "parent": curr_node,
  #       }

  #       curr_node["children"].extend([child])
    
  #   return curr_node

  def backup(self, curr_node, value, c=1):
    while curr_node["parent"] is not None:
      curr_node["N"] += 1.
      curr_node["W"] += value
      curr_node["Q"] = curr_node["W"]/curr_node["N"]
      curr_node["U"] = c*curr_node["P"]*tf.log(curr_node["parent"]["N"])/(1 + curr_node["N"])
      curr_node["uct"] = curr_node["Q"] + curr_node["U"]

      curr_node = curr_node["parent"]

    return curr_node

  # def backup(self, curr_node, value):
  #   while curr_node["parent"] is not None:
  #     curr_node["N"] += 1
  #     curr_node["W"] += value
  #     curr_node["Q"] = curr_node["W"]/curr_node["N"]
  #     curr_node["U"] = c*curr_node["P"]*np.log(curr_node["parent"]["N"])/(1 + curr_node["N"])
  #     curr_node["uct"] = Q + U

  #     curr_node = curr_node["parent"]

  #   return curr_node

  def _build_sampler(self, visits=False):
    """Build the sampler ops and the log_prob ops."""

    print "-" * 80
    print "Build controller sampler"
    anchors = []
    anchors_w_1 = []

    if self.root_node is not None:
      self.root_node["itrs"] += 1

    curr_node = self.root_node

    arc_seq = []
    entropys = []
    log_probs = []
    skip_count = []
    skip_penaltys = []

    all_h = []

    prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
              xrange(self.lstm_num_layers)]
    prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
              xrange(self.lstm_num_layers)]
    inputs = self.g_emb
    skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
                               dtype=tf.float32)
    for layer_id in xrange(self.num_layers):
      if self.search_whole_channels:
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        prev_c, prev_h = next_c, next_h
        logit = tf.matmul(next_h[-1], self.w_soft)
        all_h.append(tf.stop_gradient(next_h[-1]))
        
        curr_node, branch_id = self.single_select(curr_node, visits)
        if branch_id is None:
          all_h = tf.concat(all_h, axis=0)
          value = tf.matmul(all_h, self.w_critic)
          probas = tf.nn.softmax(logit)
          curr_node = self.expand(curr_node, logit)
          root_node = self.backup(curr_node, value)
          return
        if self.temperature is not None:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * tf.tanh(logit)
        if self.search_for == "macro" or self.search_for == "branch":
          # branch_id = tf.multinomial(logit, 1)
          # branch_id = tf.to_int32(branch_id)
          branch_id = tf.reshape(branch_id, [1])
        elif self.search_for == "connection":
          branch_id = tf.constant([0], dtype=tf.int32)
        else:
          raise ValueError("Unknown search_for {}".format(self.search_for))
        arc_seq.append(branch_id)
        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logit, labels=branch_id)
        log_probs.append(log_prob)
        entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
        entropys.append(entropy)
        inputs = tf.nn.embedding_lookup(self.w_emb, branch_id)
      else:
        for branch_id in xrange(self.num_branches):
          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = tf.matmul(next_h[-1], self.w_soft["start"][branch_id])
          all_h.append(tf.stop_gradient(next_h[-1]))
          
          curr_node, start = self.single_select(curr_node, visits)
          if start is None:
            all_h = tf.concat(all_h, axis=0)
            value = tf.matmul(all_h, self.w_critic)
            probas = tf.nn.softmax(logit)
            curr_node = self.expand(curr_node, logit)
            root_node = self.backup(curr_node, value)
            return
          if self.temperature is not None:
            logit /= self.temperature
          if self.tanh_constant is not None:
            logit = self.tanh_constant * tf.tanh(logit)
          # start = tf.multinomial(logit, 1)
          # start = tf.to_int32(start)
          start = tf.reshape(start, [1])
          arc_seq.append(start)
          log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=start)
          log_probs.append(log_prob)
          entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
          entropys.append(entropy)
          inputs = tf.nn.embedding_lookup(self.w_emb["start"][branch_id], start)

          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = tf.matmul(next_h[-1], self.w_soft["count"][branch_id])
          all_h.append(tf.stop_gradient(next_h[-1]))
          curr_node, count = self.single_select(curr_node, visits)
          if count is None:
            all_h = tf.concat(all_h, axis=0)
            value = tf.matmul(all_h, self.w_critic)
            curr_node = self.expand(curr_node, logit)
            root_node = self.backup(curr_node, value)
            return
          if self.temperature is not None:
            logit /= self.temperature
          if self.tanh_constant is not None:
            logit = self.tanh_constant * tf.tanh(logit)
          mask = tf.range(0, limit=self.out_filters-1, delta=1, dtype=tf.int32)
          mask = tf.reshape(mask, [1, self.out_filters - 1])
          mask = tf.less_equal(mask, self.out_filters-1 - start)
          logit = tf.where(mask, x=logit, y=tf.fill(tf.shape(logit), -np.inf))
          # count = tf.multinomial(logit, 1)
          # count = tf.to_int32(count)
          count = tf.reshape(count, [1])
          arc_seq.append(count + 1)
          log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=count)
          log_probs.append(log_prob)
          entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
          entropys.append(entropy)
          inputs = tf.nn.embedding_lookup(self.w_emb["count"][branch_id], count)

      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h

      if layer_id > 0:
        query = tf.concat(anchors_w_1, axis=0)
        query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
        query = tf.matmul(query, self.v_attn)
        logit = tf.concat([-query, query], axis=1)
        if self.temperature is not None:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * tf.tanh(logit)

        skip = tf.multinomial(logit, 1)
        skip = tf.to_int32(skip)
        skip = tf.reshape(skip, [layer_id])
        arc_seq.append(skip)

        skip_prob = tf.sigmoid(logit)
        kl = skip_prob * tf.log(skip_prob / skip_targets)
        kl = tf.reduce_sum(kl)
        skip_penaltys.append(kl)

        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logit, labels=skip)
        log_probs.append(tf.reduce_sum(log_prob, keep_dims=True))

        entropy = tf.stop_gradient(
          tf.reduce_sum(log_prob * tf.exp(-log_prob), keep_dims=True))
        entropys.append(entropy)

        skip = tf.to_float(skip)
        skip = tf.reshape(skip, [1, layer_id])
        skip_count.append(tf.reduce_sum(skip))
        inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
        inputs /= (1.0 + tf.reduce_sum(skip))
      else:
        inputs = self.g_emb

      anchors.append(next_h[-1])
      anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))

    arc_seq = tf.concat(arc_seq, axis=0)
    self.sample_arc = tf.reshape(arc_seq, [-1])

    entropys = tf.stack(entropys)
    self.sample_entropy = tf.reduce_sum(entropys)

    log_probs = tf.stack(log_probs)
    self.sample_log_prob = tf.reduce_sum(log_probs)

    skip_count = tf.stack(skip_count)
    self.skip_count = tf.reduce_sum(skip_count)

    skip_penaltys = tf.stack(skip_penaltys)
    self.skip_penaltys = tf.reduce_mean(skip_penaltys)

    self.all_h = all_h

  def build_trainer(self, child_model):
    child_model.build_valid_rl()
    self.valid_acc = (tf.to_float(child_model.valid_shuffle_acc) /
                      tf.to_float(child_model.batch_size))
    self.reward = self.valid_acc

    all_h = tf.concat(self.all_h, axis=0)
    value_function = tf.matmul(all_h, self.w_critic)
    advantage = value_function - self.reward
    critic_loss = tf.reduce_sum(advantage ** 2)

    critic_train_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="critic_train_step")
    critic_train_op, _, _, _ = get_train_ops(
      critic_loss,
      [self.w_critic],
      critic_train_step,
      clip_mode=None,
      lr_init=1e-3,
      lr_dec_start=0,
      lr_dec_every=int(1e9),
      optim_algo="adam",
      sync_replicas=False)

    normalize = tf.to_float(self.num_layers * (self.num_layers - 1) / 2)
    self.skip_rate = tf.to_float(self.skip_count) / normalize

    if self.entropy_weight is not None:
      self.reward += self.entropy_weight * self.sample_entropy

    self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
    
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    baseline_update = tf.assign_sub(
      self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

    with tf.control_dependencies([baseline_update]):
      self.reward = tf.identity(self.reward)

    # self.loss = self.sample_log_prob * (self.reward - self.baseline)
    search_probs = np.array([child["N"] for child in self.root_node["children"]])
    search_probs /= search_probs.sum()
    search_probs = np.expand_dims(search_probs, -1)
    search_probs = tf.convert_to_tensor(search_probs, dtype=tf.float32)
    self.sample_log_prob = tf.expand_dims(self.sample_log_prob, axis=0)
    self.loss = tf.matmul(search_probs, self.sample_log_prob)
    if self.skip_weight is not None:
      self.loss += self.skip_weight * self.skip_penaltys

    self.train_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="train_step")
    tf_variables = [var
        for var in tf.trainable_variables() if var.name.startswith(self.name)]
    print "-" * 80
    for var in tf_variables:
      print var

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.train_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      sync_replicas=self.sync_replicas,
      num_aggregate=self.num_aggregate,
      num_replicas=self.num_replicas)

