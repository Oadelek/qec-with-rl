import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add, Conv1D, GlobalAveragePooling1D, Concatenate, Dropout, BatchNormalization, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from typing import Tuple, List, Optional

# --- Enhanced Hyperparameters ---
class Hyperparameters:
    def __init__(self):
        self.CODE_DISTANCE = 3
        self.ERROR_PROBABILITY = 0.05
        self.MEASUREMENT_ERROR_PROBABILITY = 0.01
        self.MAX_CORRECTION_ROUNDS = 3
        self.REPLAY_BUFFER_SIZE = 50000
        self.BATCH_SIZE = 64
        self.GAMMA = 0.95
        self.EPSILON_START = 1.0
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY_STEPS = 30000
        self.LEARNING_RATE_START = 0.00005
        self.LEARNING_RATE_DECAY_STEPS = 20000
        self.LEARNING_RATE_END = 0.00001
        self.TARGET_NETWORK_UPDATE_FREQ = 1000
        self.TAU = 0.005
        self.HIDDEN_UNITS = [256, 128]
        self.USE_CONV_LAYERS = True
        self.DROPOUT_RATE = 0.1
        self.USE_BATCH_NORM = True
        self.NUM_EPISODES = 30000
        self.PRINT_EVERY_EPISODES = 200
        self.SAVE_MODEL_EVERY_EPISODES = 2500
        self.MODEL_SAVE_DIR = "super_enhanced_dqn_qec_model"
        self.VERBOSE_LOGGING_EPISODE_INTERVAL = 2000
        self.USE_PRIORITIZED_REPLAY = False
        self.USE_DOUBLE_DQN = True
        self.USE_SYNDROME_HISTORY = True
        self.SYNDROME_HISTORY_LENGTH = 3

HPARAMS = Hyperparameters()

# --- Surface Code Generator ---
class SurfaceCodeGenerator:
    @staticmethod
    def generate_code_properties(distance: int, rotated_d3: bool = True) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        num_data_qubits = distance * distance
        if distance == 3 and rotated_d3:
            z_stabilizers = np.array([
                [1,1,0,1,1,0,0,0,0],[0,1,1,0,1,1,0,0,0],
                [0,0,0,1,1,0,1,1,0],[0,0,0,0,1,1,0,1,1]], dtype=int)
            x_stabilizers = np.array([ # Example, ensure consistency if used
                [1,0,1,1,0,0,0,0,0], [0,1,1,0,1,0,0,0,0],
                [0,0,0,1,0,1,1,0,1], [0,0,0,0,1,1,0,1,1]], dtype=int)
            logical_x = [np.array([1,0,0,1,0,0,1,0,0], dtype=int)]
            logical_z = [np.array([1,1,1,0,0,0,0,0,0], dtype=int)]
        elif distance == 5: # Standard (non-rotated) planar code for d=5
            z_stabilizers_list = []
            for r in range(distance - 1):
                for c in range(distance - 1):
                    stab = np.zeros(num_data_qubits, dtype=int)
                    q_tl = r * distance + c
                    q_tr = r * distance + (c + 1)
                    q_bl = (r + 1) * distance + c
                    q_br = (r + 1) * distance + (c + 1)
                    stab[[q_tl, q_tr, q_bl, q_br]] = 1
                    z_stabilizers_list.append(stab)
            z_stabilizers = np.array(z_stabilizers_list)
            x_stabilizers = np.array([]) # Not generating full X stabs for brevity
            logical_x = [np.zeros(num_data_qubits, dtype=int)]
            logical_x[0][0:num_data_qubits:distance] = 1
            logical_z = [np.zeros(num_data_qubits, dtype=int)]
            logical_z[0][0:distance] = 1
        else:
            raise ValueError(f"Code distance {distance} not properly implemented in generator.")
        return z_stabilizers, x_stabilizers, logical_x, logical_z

# --- Enhanced Network Architecture ---
class EnhancedQNetwork:
    def __init__(self, state_size: int, action_size: int, hparams: Hyperparameters,
                 learning_rate_schedule_object: tf.keras.optimizers.schedules.LearningRateSchedule,
                 model_name="q_network"):
        self.state_size = state_size
        self.action_size = action_size
        self.hparams = hparams
        self.model_name = model_name
        self.learning_rate_schedule_object = learning_rate_schedule_object
        self.model = self._build_enhanced_model()

    def _build_enhanced_model(self):
        inputs = Input(shape=(self.state_size,), name=f"{self.model_name}_input")
        x = inputs
        if self.hparams.USE_CONV_LAYERS and self.state_size >= (self.hparams.SYNDROME_HISTORY_LENGTH if self.hparams.USE_SYNDROME_HISTORY else 1) * 3:
            num_stabs_features = self.state_size // (self.hparams.SYNDROME_HISTORY_LENGTH if self.hparams.USE_SYNDROME_HISTORY else 1)
            if self.hparams.USE_SYNDROME_HISTORY and self.hparams.SYNDROME_HISTORY_LENGTH > 1:
                reshaped_input = Reshape((self.hparams.SYNDROME_HISTORY_LENGTH, num_stabs_features))(x)
                conv_layer = Conv1D(32, kernel_size=min(3, self.hparams.SYNDROME_HISTORY_LENGTH), activation='relu', padding='same')(reshaped_input)
                if self.hparams.USE_BATCH_NORM: conv_layer = BatchNormalization()(conv_layer)
                conv_layer = Flatten()(conv_layer)
            elif num_stabs_features >=3 :
                reshaped_input = Reshape((num_stabs_features, 1))(x)
                conv_layer = Conv1D(32, kernel_size=3, activation='relu', padding='same')(reshaped_input)
                if self.hparams.USE_BATCH_NORM: conv_layer = BatchNormalization()(conv_layer)
                conv_layer = GlobalAveragePooling1D()(conv_layer)
            else: conv_layer = Flatten()(x)
            x = Concatenate()([Flatten()(inputs), conv_layer])
        else: x = Flatten()(inputs)
        for units in self.hparams.HIDDEN_UNITS:
            dense_layer = Dense(units, activation='relu')(x)
            if self.hparams.USE_BATCH_NORM: dense_layer = BatchNormalization()(dense_layer)
            if self.hparams.DROPOUT_RATE > 0: dense_layer = Dropout(self.hparams.DROPOUT_RATE)(dense_layer)
            x = dense_layer
        value_stream = Dense(self.hparams.HIDDEN_UNITS[-1]//2 if self.hparams.HIDDEN_UNITS else 64, activation='relu')(x)
        value = Dense(1, activation='linear', name='value')(value_stream)
        advantage_stream = Dense(self.hparams.HIDDEN_UNITS[-1]//2 if self.hparams.HIDDEN_UNITS else 64, activation='relu')(x)
        advantage = Dense(self.action_size, activation='linear', name='advantage')(advantage_stream)
        def dueling_aggregator(streams): v, adv = streams; return v + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))
        q_values = Lambda(dueling_aggregator, name='q_values')([value, advantage])
        model = Model(inputs=inputs, outputs=q_values, name=self.model_name)
        optimizer = Adam(learning_rate=self.learning_rate_schedule_object, clipnorm=1.0) # Pass schedule object
        model.compile(loss=Huber(delta=1.0), optimizer=optimizer)
        return model

    def predict(self, state: np.ndarray) -> np.ndarray:
        if len(state.shape) == 1: state = np.expand_dims(state, axis=0)
        return self.model.predict(state, verbose=0)

    def update(self, states: np.ndarray, targets_full: np.ndarray) -> float: # Removed LR param
        history = self.model.fit(states, targets_full, batch_size=len(states), epochs=1, verbose=0)
        return history.history['loss'][0]

    def save(self, filepath: str): self.model.save_weights(filepath)
    def load(self, filepath: str): self.model.load_weights(filepath)

# --- Enhanced Surface Code Environment ---
class EnhancedSurfaceCodeEnv:
    def __init__(self, hparams: Hyperparameters):
        self.hparams = hparams; self.code_distance = hparams.CODE_DISTANCE
        self.data_qubits = self.code_distance * self.code_distance
        self.z_stabilizers, _, self.logical_x_ops, _ = \
            SurfaceCodeGenerator.generate_code_properties(self.code_distance, rotated_d3=(self.code_distance==3))
        self.num_z_stabilizers = self.z_stabilizers.shape[0]
        self.current_x_errors = np.zeros(self.data_qubits, dtype=int)
        self.current_round = 0
        self.syndrome_history = deque(maxlen=self.hparams.SYNDROME_HISTORY_LENGTH if self.hparams.USE_SYNDROME_HISTORY else 1)
        self.logical_error_in_episode = False
    def _compute_syndrome(self, error_pattern: np.ndarray) -> np.ndarray:
        if self.z_stabilizers.shape[1] != error_pattern.shape[0]:
             raise ValueError(f"Shape mismatch: Z_stabs {self.z_stabilizers.shape}, error {error_pattern.shape}")
        return np.mod(self.z_stabilizers @ error_pattern, 2)
    def _add_measurement_errors(self, syndrome: np.ndarray) -> np.ndarray:
        flips = np.random.binomial(1, self.hparams.MEASUREMENT_ERROR_PROBABILITY, len(syndrome))
        return np.mod(syndrome + flips, 2)
    def _get_current_state_representation(self) -> np.ndarray:
        if self.hparams.USE_SYNDROME_HISTORY:
            flat_history = []; num_to_fill = self.hparams.SYNDROME_HISTORY_LENGTH
            padding_syndrome = self.syndrome_history[0] if self.syndrome_history else np.zeros(self.num_z_stabilizers, dtype=np.float32)
            for _ in range(num_to_fill - len(self.syndrome_history)): flat_history.extend(padding_syndrome)
            for s in self.syndrome_history: flat_history.extend(s)
            return np.array(flat_history, dtype=np.float32)
        else:
            return self.syndrome_history[-1].astype(np.float32) if self.syndrome_history else np.zeros(self.num_z_stabilizers, dtype=np.float32)
    def reset(self, verbose_episode: bool = False) -> np.ndarray:
        self.current_x_errors = np.random.binomial(1, self.hparams.ERROR_PROBABILITY, self.data_qubits)
        self.current_round = 0; self.syndrome_history.clear(); self.logical_error_in_episode = False
        true_syndrome = self._compute_syndrome(self.current_x_errors)
        observed_syndrome = self._add_measurement_errors(true_syndrome)
        for _ in range(self.hparams.SYNDROME_HISTORY_LENGTH if self.hparams.USE_SYNDROME_HISTORY else 1):
            self.syndrome_history.append(observed_syndrome.copy())
        if verbose_episode:
            print(f"    ENV RESET: Initial X Errors ({np.sum(self.current_x_errors)}): {self.current_x_errors.tolist()}")
            print(f"    ENV RESET: True Syndrome: {true_syndrome.tolist()}")
            print(f"    ENV RESET: Observed Syndrome (State): {observed_syndrome.tolist()}")
        return self._get_current_state_representation()
    def step(self, action_qubit_index: int, verbose_episode: bool = False) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_round += 1
        correction_pattern = np.zeros(self.data_qubits, dtype=int); correction_pattern[action_qubit_index] = 1
        self.current_x_errors = np.mod(self.current_x_errors + correction_pattern, 2)
        true_residual_syndrome = self._compute_syndrome(self.current_x_errors)
        observed_residual_syndrome = self._add_measurement_errors(true_residual_syndrome)
        if self.hparams.USE_SYNDROME_HISTORY: self.syndrome_history.append(observed_residual_syndrome.copy())
        else: self.syndrome_history.clear(); self.syndrome_history.append(observed_residual_syndrome.copy())
        current_logical_error_this_step = self._has_logical_x_error(self.current_x_errors)
        if current_logical_error_this_step: self.logical_error_in_episode = True
        reward = self._calculate_reward(self.current_x_errors, observed_residual_syndrome, current_logical_error_this_step, correction_pattern)
        done_this_round = self.logical_error_in_episode or np.all(observed_residual_syndrome == 0) or self.current_round >= self.hparams.MAX_CORRECTION_ROUNDS
        next_state_representation = self._get_current_state_representation()
        info = {'true_syndrome': true_residual_syndrome, 'observed_syndrome': observed_residual_syndrome,
                'logical_error_this_step': current_logical_error_this_step, 'logical_error_in_episode': self.logical_error_in_episode}
        if verbose_episode:
            print(f"    ENV ROUND {self.current_round}: Action (flip qubit {action_qubit_index})")
            print(f"    ENV ROUND {self.current_round}: Residual Errors ({np.sum(self.current_x_errors)})")
            print(f"    ENV ROUND {self.current_round}: True Res Synd: {true_residual_syndrome.tolist()}")
            print(f"    ENV ROUND {self.current_round}: Obs Res Synd: {observed_residual_syndrome.tolist()}")
            print(f"    ENV ROUND {self.current_round}: Log Err (step): {current_logical_error_this_step}, Reward: {reward:.3f}, Done: {done_this_round}")
        return next_state_representation, reward, done_this_round, info
    def _calculate_reward(self, residual_errors: np.ndarray, observed_syndrome: np.ndarray, logical_error_this_step: bool, action_taken: np.ndarray) -> float:
        if logical_error_this_step: return -1.0
        syndrome_sum = np.sum(observed_syndrome)
        if syndrome_sum == 0: return 1.0 if np.sum(residual_errors) == 0 else 0.8
        return -0.1 * (syndrome_sum / self.num_z_stabilizers)
    def _is_stabilizer_element(self, error_pattern: np.ndarray) -> bool:
        if not np.any(error_pattern): return True
        return np.all(self._compute_syndrome(error_pattern) == 0)
    def _has_logical_x_error(self, x_error_pattern: np.ndarray) -> bool:
        if not np.any(x_error_pattern): return False
        if self._is_stabilizer_element(x_error_pattern): return False
        for log_x_op_def in self.logical_x_ops:
            if len(log_x_op_def) != len(x_error_pattern): continue
            diff_pattern = np.mod(x_error_pattern + log_x_op_def, 2)
            if self._is_stabilizer_element(diff_pattern): return True
        return False

# --- Enhanced DQN Agent ---
class EnhancedDQNAgent:
    def __init__(self, state_size: int, action_size: int, hparams: Hyperparameters):
        self.state_size = state_size; self.action_size = action_size; self.hparams = hparams
        self.lr_schedule_fn = ExponentialDecay(
            initial_learning_rate=hparams.LEARNING_RATE_START,
            decay_steps=hparams.LEARNING_RATE_DECAY_STEPS,
            decay_rate=(hparams.LEARNING_RATE_END / hparams.LEARNING_RATE_START)**(1.0 / ( (hparams.NUM_EPISODES * hparams.MAX_CORRECTION_ROUNDS) / hparams.LEARNING_RATE_DECAY_STEPS if hparams.LEARNING_RATE_DECAY_STEPS > 0 else 1.0) + 1e-6))

        self.current_learning_rate_for_logging = hparams.LEARNING_RATE_START # For logging
        self.q_network = EnhancedQNetwork(state_size, action_size, hparams, self.lr_schedule_fn, "q_main")
        self.target_network = EnhancedQNetwork(state_size, action_size, hparams, self.lr_schedule_fn, "q_target")
        self.update_target_network(tau=1.0)
        self.replay_buffer = deque(maxlen=hparams.REPLAY_BUFFER_SIZE)
        self.epsilon = hparams.EPSILON_START
        self.epsilon_decay_factor = (hparams.EPSILON_MIN / hparams.EPSILON_START)**(1.0 / (hparams.EPSILON_DECAY_STEPS))
        self.env_steps_done = 0; self.training_steps_done = 0
    def update_target_network(self, tau: Optional[float] = None):
        if tau is None: tau = self.hparams.TAU
        if tau == 1.0: self.target_network.model.set_weights(self.q_network.model.get_weights())
        else:
            q_weights=self.q_network.model.get_weights(); target_weights=self.target_network.model.get_weights()
            new_weights=[tau*q_w+(1-tau)*t_w for q_w,t_w in zip(q_weights,target_weights)]
            self.target_network.model.set_weights(new_weights)
    def remember(self, state:np.ndarray, action_idx:int, reward:float, next_state:np.ndarray, done_this_round:bool):
        self.replay_buffer.append((state,action_idx,reward,next_state,done_this_round))
    def act(self, state: np.ndarray, use_exploration: bool = True) -> int:
        self.env_steps_done +=1
        if use_exploration and random.random() <= self.epsilon: return random.randrange(self.action_size)
        q_values = self.q_network.predict(state); return np.argmax(q_values[0])
    def replay(self) -> float:
        if len(self.replay_buffer) < self.hparams.BATCH_SIZE: return 0.0
        minibatch=random.sample(self.replay_buffer,self.hparams.BATCH_SIZE)
        states=np.array([exp[0] for exp in minibatch]);action_indices=np.array([exp[1] for exp in minibatch])
        rewards=np.array([exp[2] for exp in minibatch]);next_states=np.array([exp[3] for exp in minibatch])
        dones_this_round=np.array([exp[4] for exp in minibatch])
        current_q_values_s=self.q_network.predict(states);targets_q_s_a=np.copy(current_q_values_s)
        q_values_s_prime_target_net=self.target_network.predict(next_states)
        if self.hparams.USE_DOUBLE_DQN:
            q_values_s_prime_main_net=self.q_network.predict(next_states)
            best_actions_s_prime=np.argmax(q_values_s_prime_main_net,axis=1)
            max_q_s_prime=q_values_s_prime_target_net[np.arange(self.hparams.BATCH_SIZE),best_actions_s_prime]
        else: max_q_s_prime=np.max(q_values_s_prime_target_net,axis=1)
        for i in range(self.hparams.BATCH_SIZE):
            if dones_this_round[i]: targets_q_s_a[i,action_indices[i]]=rewards[i]
            else: targets_q_s_a[i,action_indices[i]]=rewards[i]+self.hparams.GAMMA*max_q_s_prime[i]

        loss=self.q_network.update(states,targets_q_s_a) # LR not passed, optimizer uses its schedule

        self.current_learning_rate_for_logging = self.lr_schedule_fn(self.training_steps_done)
        if isinstance(self.current_learning_rate_for_logging,tf.Tensor):
            self.current_learning_rate_for_logging=float(self.current_learning_rate_for_logging.numpy())
        self.training_steps_done+=1; return loss
    def decay_epsilon(self):
        if self.epsilon > self.hparams.EPSILON_MIN:
            if self.env_steps_done <= self.hparams.EPSILON_DECAY_STEPS :
                 self.epsilon = self.hparams.EPSILON_START * (self.epsilon_decay_factor ** self.env_steps_done)
            else: self.epsilon = self.hparams.EPSILON_MIN
            self.epsilon = max(self.hparams.EPSILON_MIN, self.epsilon)
    def save(self, directory: str, episode: int):
        if not os.path.exists(directory): os.makedirs(directory)
        self.q_network.save(os.path.join(directory,f"q_network_ep{episode}.weights.h5"))
        agent_state={'episode':episode,'epsilon':self.epsilon,'training_steps_done':self.training_steps_done,
                       'env_steps_done':self.env_steps_done,
                       'current_lr_for_logging': float(self.current_learning_rate_for_logging),
                       'hparams':vars(self.hparams)}
        with open(os.path.join(directory,f"agent_state_ep{episode}.json"),'w') as f: json.dump(agent_state,f,indent=4)
        print(f"Saved model and agent state at episode {episode}")
    def load(self, directory: str, episode: int):
        self.q_network.load(os.path.join(directory,f"q_network_ep{episode}.weights.h5"))
        self.update_target_network(tau=1.0)
        with open(os.path.join(directory,f"agent_state_ep{episode}.json"),'r') as f: agent_state=json.load(f)
        self.epsilon=agent_state['epsilon'];self.training_steps_done=agent_state.get('training_steps_done',0)
        self.env_steps_done=agent_state.get('env_steps_done',0)
        self.current_learning_rate_for_logging = self.lr_schedule_fn(self.training_steps_done)
        if isinstance(self.current_learning_rate_for_logging,tf.Tensor):
            self.current_learning_rate_for_logging=float(self.current_learning_rate_for_logging.numpy())
        print(f"Loaded model from ep {episode}, epsilon: {self.epsilon:.4f}, training_steps: {self.training_steps_done}, env_steps: {self.env_steps_done}, logged LR: {self.current_learning_rate_for_logging:.6e}")

# --- Main Training Function ---
def train_enhanced_qec_agent(hparams: Hyperparameters):
    env = EnhancedSurfaceCodeEnv(hparams)
    state_size_per_syndrome_reading = env.num_z_stabilizers
    state_size_for_agent = state_size_per_syndrome_reading * (hparams.SYNDROME_HISTORY_LENGTH if hparams.USE_SYNDROME_HISTORY else 1)
    action_size_for_agent = env.data_qubits
    agent = EnhancedDQNAgent(state_size_for_agent, action_size_for_agent, hparams)
    metrics = {'ep_rewards':[],'round_losses':[],'ep_epsilon':[],'ep_logical_errors':[],'ep_rounds':[], 'learning_rates':[]}
    print(f"Starting training: d={hparams.CODE_DISTANCE}, P_err={hparams.ERROR_PROBABILITY}, P_meas={hparams.MEASUREMENT_ERROR_PROBABILITY}")
    print(f"State size for agent: {state_size_for_agent}, Action size: {action_size_for_agent} (flips one qubit)")
    print(f"Max rounds per episode: {hparams.MAX_CORRECTION_ROUNDS}, Gamma: {hparams.GAMMA}")
    for episode in tqdm(range(hparams.NUM_EPISODES), desc="Training Episodes"):
        verbose_episode = (episode+1)%hparams.VERBOSE_LOGGING_EPISODE_INTERVAL==0
        if verbose_episode: tqdm.write(f"\n--- Episode {episode+1} (Verbose) ---")
        current_state_repr = env.reset(verbose_episode=verbose_episode); episode_total_reward = 0
        for round_num in range(hparams.MAX_CORRECTION_ROUNDS):
            action_idx = agent.act(current_state_repr); agent.decay_epsilon()
            next_state_repr,round_reward,round_done,info = env.step(action_idx,verbose_episode=verbose_episode)
            agent.remember(current_state_repr,action_idx,round_reward,next_state_repr,round_done)
            loss_this_replay=agent.replay();
            if loss_this_replay > 0: metrics['round_losses'].append(loss_this_replay)
            metrics['learning_rates'].append(float(agent.current_learning_rate_for_logging))
            episode_total_reward+=round_reward; current_state_repr=next_state_repr
            if round_done: break
        metrics['ep_rewards'].append(episode_total_reward);metrics['ep_epsilon'].append(agent.epsilon)
        metrics['ep_logical_errors'].append(1 if env.logical_error_in_episode else 0)
        metrics['ep_rounds'].append(env.current_round)
        if (episode+1)%hparams.TARGET_NETWORK_UPDATE_FREQ==0: agent.update_target_network()
        if (episode+1)%hparams.PRINT_EVERY_EPISODES==0:
            avg_ep_reward=np.mean(metrics['ep_rewards'][-hparams.PRINT_EVERY_EPISODES:]) if metrics['ep_rewards'] else 0
            num_replays_in_window=sum(metrics['ep_rounds'][-hparams.PRINT_EVERY_EPISODES:]) if metrics['ep_rounds'] else hparams.PRINT_EVERY_EPISODES
            avg_round_loss=np.mean(metrics['round_losses'][-int(num_replays_in_window):]) if metrics['round_losses'] else 0
            avg_ep_logical_errors=np.mean(metrics['ep_logical_errors'][-hparams.PRINT_EVERY_EPISODES:])*100 if metrics['ep_logical_errors'] else 0
            avg_ep_rounds=np.mean(metrics['ep_rounds'][-hparams.PRINT_EVERY_EPISODES:]) if metrics['ep_rounds'] else 0
            current_lr_to_log = agent.current_learning_rate_for_logging
            tqdm.write(f"Ep {episode+1}/{hparams.NUM_EPISODES} | Avg Reward: {avg_ep_reward:.2f} | Avg Rounds: {avg_ep_rounds:.2f} | Log.Err %: {avg_ep_logical_errors:.2f}% | Avg Loss: {avg_round_loss:.4f} | Epsilon: {agent.epsilon:.4f} | LR: {current_lr_to_log:.6e}")
        if (episode+1)%hparams.SAVE_MODEL_EVERY_EPISODES==0: agent.save(hparams.MODEL_SAVE_DIR,episode+1)
    print("\n--- Training Finished ---"); agent.save(hparams.MODEL_SAVE_DIR,hparams.NUM_EPISODES)
    plot_training_results(metrics,hparams); test_qec_agent(agent,hparams,num_test_episodes=500)

def plot_training_results(metrics,hparams):
    fig,axs=plt.subplots(5,1,figsize=(12,24),sharex=True);window=hparams.PRINT_EVERY_EPISODES
    axs[0].plot(metrics['ep_rewards'], alpha=0.3, label='Per-Episode Reward')
    if len(metrics['ep_rewards'])>=window:axs[0].plot(np.arange(window-1, len(metrics['ep_rewards'])),np.convolve(metrics['ep_rewards'], np.ones(window)/window, mode='valid'), label=f'MovAvg Ep Reward (window {window})')
    axs[0].set_title(f'Training Rewards (d={hparams.CODE_DISTANCE}, P_err={hparams.ERROR_PROBABILITY})');axs[0].set_ylabel('Total Episode Reward');axs[0].legend();axs[0].grid(True)
    valid_losses=[l for l in metrics['round_losses'] if l is not None and l>0]
    axs[1].plot(valid_losses, alpha=0.2, label='Per-Replay Loss')
    if valid_losses and len(valid_losses)>=window : axs[1].plot(np.arange(window-1, len(valid_losses)), np.convolve(valid_losses, np.ones(window)/window, mode='valid'), label=f'MovAvg Loss (window {window})')
    axs[1].set_title('Training Loss (Huber)');axs[1].set_ylabel('Loss');axs[1].set_yscale('log');axs[1].legend();axs[1].grid(True)
    ax2_twin=axs[2].twinx();axs[2].plot(metrics['ep_epsilon'], color='blue', label='Epsilon')
    axs[2].set_ylabel('Epsilon', color='blue');axs[2].tick_params(axis='y', labelcolor='blue');axs[2].legend(loc='upper left')
    if len(metrics['ep_logical_errors'])>=window:ax2_twin.plot(np.arange(window-1, len(metrics['ep_logical_errors'])), np.convolve(metrics['ep_logical_errors'], np.ones(window)/window, mode='valid'), color='red', label=f'MovAvg Logical Error Rate (window {window})')
    ax2_twin.set_ylabel('Logical Error Rate', color='red');ax2_twin.tick_params(axis='y', labelcolor='red');ax2_twin.legend(loc='upper right');axs[2].set_title('Epsilon & Logical Error Rate');axs[2].grid(True)
    axs[3].plot(metrics['ep_rounds'], alpha=0.3, label='Rounds/Ep')
    if len(metrics['ep_rounds'])>=window:axs[3].plot(np.arange(window-1, len(metrics['ep_rounds'])), np.convolve(metrics['ep_rounds'], np.ones(window)/window, mode='valid'), label=f'MovAvg Rounds/Ep (window {window})')
    axs[3].set_title('Average Rounds per Episode');axs[3].set_ylabel('Rounds');axs[3].legend();axs[3].grid(True)
    axs[4].plot(metrics['learning_rates'], label='Learning Rate')
    axs[4].set_title('Learning Rate Decay'); axs[4].set_ylabel('Learning Rate'); axs[4].set_yscale('log')
    axs[4].set_xlabel('Training Step (Replay Call)'); axs[4].legend(); axs[4].grid(True)
    plt.tight_layout();save_path=os.path.join(hparams.MODEL_SAVE_DIR,f"training_summary_d{hparams.CODE_DISTANCE}.png")
    if not os.path.exists(hparams.MODEL_SAVE_DIR):os.makedirs(hparams.MODEL_SAVE_DIR)
    plt.savefig(save_path);print(f"Training summary plots saved to {save_path}");plt.show()

def test_qec_agent(agent:EnhancedDQNAgent,hparams:Hyperparameters,num_test_episodes:int):
    print("\n--- Testing Enhanced Learned Policy (Greedy Exploration) ---");env=EnhancedSurfaceCodeEnv(hparams)
    total_logical_errors_test=0;total_rounds_test=0;final_syndrome_weight_sum=0;successful_syndrome_clear=0
    for episode in tqdm(range(num_test_episodes),desc="Testing Progress"):
        verbose_test_episode=episode<3;current_state_repr=env.reset(verbose_episode=verbose_test_episode)
        if verbose_test_episode:tqdm.write(f"\n--- Test Episode {episode+1} ---")
        for round_num in range(hparams.MAX_CORRECTION_ROUNDS):
            action_idx=agent.act(current_state_repr,use_exploration=False)
            next_state_repr,_,round_done,info=env.step(action_idx,verbose_episode=verbose_test_episode)
            current_state_repr=next_state_repr;total_rounds_test+=1
            if round_done:
                if info['logical_error_in_episode']:total_logical_errors_test+=1
                if np.all(info['observed_syndrome']==0):successful_syndrome_clear+=1
                final_syndrome_weight_sum+=np.sum(info['observed_syndrome']);break
            elif round_num==hparams.MAX_CORRECTION_ROUNDS-1:
                 if info['logical_error_in_episode']:total_logical_errors_test+=1
                 final_syndrome_weight_sum+=np.sum(info['observed_syndrome'])
    logical_error_rate=(total_logical_errors_test/num_test_episodes)*100 if num_test_episodes>0 else 0
    avg_rounds=total_rounds_test/num_test_episodes if num_test_episodes>0 else 0
    avg_final_syndrome_weight=final_syndrome_weight_sum/num_test_episodes if num_test_episodes>0 else 0
    perc_syndrome_cleared=(successful_syndrome_clear/num_test_episodes)*100 if num_test_episodes>0 else 0
    print(f"\nTest Results ({num_test_episodes} episodes):")
    print(f"  Logical Error Rate: {logical_error_rate:.2f}% ({total_logical_errors_test}/{num_test_episodes})")
    print(f"  Average Rounds per Episode: {avg_rounds:.2f}")
    print(f"  Episodes with Syndrome Cleared: {perc_syndrome_cleared:.2f}%")
    print(f"  Average Final Syndrome Weight (observed): {avg_final_syndrome_weight:.2f}")

if __name__ == "__main__":
    train_enhanced_qec_agent(HPARAMS)