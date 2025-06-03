import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json

# --- Hyperparameters ---
class Hyperparameters:
    def __init__(self):
        # Environment
        self.CODE_DISTANCE = 3 # d=3
        self.ERROR_PROBABILITY = 0.05

        # DQN Agent
        self.REPLAY_BUFFER_SIZE = 50000
        self.BATCH_SIZE = 128 # Larger batch size can stabilize training
        self.GAMMA = 0.0  # Discount factor (0 for single-step terminal episodes)
        self.EPSILON_START = 1.0
        self.EPSILON_MIN = 0.01
        self.EPSILON_DECAY_STEPS = 15000 # Number of steps over which epsilon decays significantly
        self.LEARNING_RATE_START = 0.001
        self.LEARNING_RATE_DECAY_STEPS = 10000
        self.LEARNING_RATE_END = 0.0001 # Or use a decay rate
        self.TARGET_NETWORK_UPDATE_FREQ = 500 # Episodes
        self.TAU = 1.0 # For hard target network updates. Use < 1.0 for soft updates.

        # NN Architecture
        self.HIDDEN_UNITS = [128, 128] # More capacity

        # Training
        self.NUM_EPISODES = 30000
        self.PRINT_EVERY_EPISODES = 200
        self.SAVE_MODEL_EVERY_EPISODES = 5000
        self.MODEL_SAVE_DIR = "dqn_qec_model"

HPARAMS = Hyperparameters()

# --- QNetwork Implementation (Dueling Double DQN) ---
class QNetwork:
    def __init__(self, state_size, action_size, hidden_units, learning_rate_schedule, model_name="q_network"):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.learning_rate_schedule = learning_rate_schedule
        self.model_name = model_name
        self.model = self._build_dueling_model()

    def _build_dueling_model(self):
        inputs = Input(shape=(self.state_size,), name=f"{self.model_name}_input")

        # Common stream
        common = inputs
        for units in self.hidden_units:
            common = Dense(units, activation='relu')(common)

        # Value stream
        value_stream = Dense(self.hidden_units[-1] // 2, activation='relu')(common) # Or same size as last common
        value = Dense(1, activation='linear', name='value')(value_stream)

        # Advantage stream
        advantage_stream = Dense(self.hidden_units[-1] // 2, activation='relu')(common) # Or same size
        advantage = Dense(self.action_size, activation='linear', name='advantage')(advantage_stream)

        # Combine value and advantage streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Lambda layer for custom operation
        def dueling_aggregator(streams):
            v, adv = streams
            return v + (adv - tf.reduce_mean(adv, axis=1, keepdims=True))

        q_values = Lambda(dueling_aggregator, name='q_values')([value, advantage])
        
        model = Model(inputs=inputs, outputs=q_values, name=self.model_name)
        optimizer = Adam(learning_rate=self.learning_rate_schedule)
        model.compile(loss='huber_loss', optimizer=optimizer) # Huber loss can be more robust
        return model

    def predict(self, state):
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        return self.model.predict(state, verbose=0)

    def update(self, states, targets_full):
        history = self.model.fit(states, targets_full, batch_size=len(states), epochs=1, verbose=0)
        return history.history['loss'][0]

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)


# --- SurfaceCodeEnv Implementation (d=3 Rotated) ---
class SurfaceCodeEnv:
    def __init__(self, code_distance=3, error_prob=0.01):
        if code_distance != 3:
            raise NotImplementedError("Currently hardcoded for d=3 rotated surface code (9 data qubits)")
        
        self.code_distance = code_distance # d
        self.data_qubits = code_distance * code_distance # N = d^2
        self.error_prob = error_prob

        # For d=3 rotated surface code:
        # Data Qubit Indexing (example):
        # . 0 . 1 . 2
        # 3 . 4 . 5 .
        # . 6 . 7 . 8
        
        # Z-Stabilizers (Plaquettes - 4 for d=3)
        # S_Z0: (0,1,4,3), S_Z1: (1,2,5,4), S_Z2: (3,4,7,6), S_Z3: (4,5,8,7)
        self.z_stabilizer_gens = np.array([
            [1,1,0,1,1,0,0,0,0],
            [0,1,1,0,1,1,0,0,0],
            [0,0,0,1,1,0,1,1,0],
            [0,0,0,0,1,1,0,1,1]
        ], dtype=int)
        self.num_z_stabilizers = self.z_stabilizer_gens.shape[0]

        # X-Stabilizers (Stars - 4 for d=3)
        # S_X0: (0,3), S_X1: (1,4), S_X2: (2,5), S_X3: (6,7,3,4), S_X4: (7,8,4,5)
        # This depends on how you define stars. A common way:
        # S_X0: (0,1,3), S_X1: (1,2,4,5), S_X2: (3,4,6,7), S_X3: (4,5,7,8) for a non-rotated grid.
        # For rotated code, X stabilizers are typically 2-body or 4-body terms around vertices.
        # Let's define X-stabilizers based on common vertices for d=3 rotated.
        # Vertex between (0,3), Vertex between (1,2,4,5), Vertex between (6,7), etc.
        # This is simpler for homology check if we define them clearly.
        # Example based on vertices:
        # q0,q3 -- S_X0
        # q1,q4 -- S_X1
        # q2,q5 -- S_X2
        # q6,q7 -- S_X3
        # q0 -- (this data qubit is involved in Z0, and an X stabilizer "touching" it)
        self.x_stabilizer_gens = np.array([ # For checking Z-logical errors (not used by this X-error agent)
            [1,0,0,1,0,0,0,0,0], # Example: S_X around vertex for q0, q3
            [0,1,0,0,1,0,0,0,0], # Example: S_X around vertex for q1, q4
            [0,0,1,0,0,1,0,0,0], # Example: S_X around vertex for q2, q5
            [0,0,0,0,0,0,1,1,0]  # Example: S_X around vertex for q6, q7
        ], dtype=int) # THIS IS A SIMPLIFICATION for illustrative purposes.
                       # Real X stabilizers are more complex and complete for d=3.

        # Logical Operators (for d=3 rotated code)
        # Logical X_L (e.g., vertical string of X on physical qubits)
        self.logical_x_ops = [np.array([1,0,0,1,0,0,1,0,0], dtype=int)] # X on q0,q3,q6
        # Logical Z_L (e.g., horizontal string of Z on physical qubits)
        self.logical_z_ops = [np.array([1,1,1,0,0,0,0,0,0], dtype=int)] # Z on q0,q1,q2

        self.current_x_errors = np.zeros(self.data_qubits, dtype=int)

    def _compute_syndrome(self, error_pattern):
        # Z-stabilizers measure X errors
        return np.mod(self.z_stabilizer_gens @ error_pattern, 2)

    def reset(self):
        self.current_x_errors = np.random.binomial(1, self.error_prob, self.data_qubits)
        syndrome = self._compute_syndrome(self.current_x_errors)
        return syndrome.astype(np.float32)

    def step(self, action_binary_vector):
        # action_binary_vector: X-Pauli corrections to apply
        if not isinstance(action_binary_vector, np.ndarray):
            action_binary_vector = np.array(action_binary_vector)
        
        residual_x_errors = np.mod(self.current_x_errors + action_binary_vector, 2)
        
        logical_x_error_occurred = self._has_logical_x_error(residual_x_errors)
        
        reward = 1.0 if not logical_x_error_occurred else -1.0
        done = True
        next_state = None # Terminal state
        
        return next_state, reward, done

    def _is_stabilizer_element(self, error_pattern, stabilizer_gens):
        """Checks if error_pattern is in the span of stabilizer_gens (i.e., a sum of them)."""
        # This can be done by checking if error_pattern has a trivial syndrome with THE DUAL stabilizers.
        # Or, more directly for small codes: try to find coefficients.
        # A simpler check: if applying the error pattern results in a zero syndrome *with respect to the same stabilizer type*,
        # AND it's not the all-zero pattern (unless we are checking for zero).
        if not np.any(error_pattern): # All-zero pattern is trivially a stabilizer (0 * S_i)
            return True
        
        # Check if error_pattern itself has a zero syndrome w.r.t the Z stabilizers
        # (This means it's a logical Z or a Z stabilizer, but we're dealing with X errors here)
        # For X errors, if an error pattern P is a Z-stabilizer or sum of Z-stabilizers,
        # its Z-syndrome (Z_gens @ P) will be zero.
        syndrome_of_pattern = np.mod(self.z_stabilizer_gens @ error_pattern, 2)
        return np.all(syndrome_of_pattern == 0)


    def _has_logical_x_error(self, x_error_pattern):
        """
        Checks if the x_error_pattern results in a logical X error.
        This means: x_error_pattern = L_X + S_Z_sum, where L_X is a logical X operator,
        and S_Z_sum is a sum of Z-stabilizer generators.
        Equivalently: x_error_pattern + L_X is in the Z-stabilizer group.
        """
        if not np.any(x_error_pattern): # No errors means no logical error
            return False

        # Check if x_error_pattern itself is equivalent to a trivial error (a sum of Z stabilizers)
        # If so, it's corrected (no logical error).
        if self._is_stabilizer_element(x_error_pattern, self.z_stabilizer_gens):
            return False # Corrected to a stabilizer state (or the ground state)

        # Now check if it's equivalent to a logical X operator
        for log_x in self.logical_x_ops:
            # Calculate pattern_sum_log_x = x_error_pattern + log_x (mod 2)
            pattern_sum_log_x = np.mod(x_error_pattern + log_x, 2)
            # If pattern_sum_log_x is a Z-stabilizer (or sum of them), then x_error_pattern is a logical X error.
            if self._is_stabilizer_element(pattern_sum_log_x, self.z_stabilizer_gens):
                return True
        return False

# --- DQNAgent Implementation ---
class DQNAgent:
    def __init__(self, state_size, action_size, hparams: Hyperparameters):
        self.state_size = state_size
        self.action_size = action_size
        self.hparams = hparams

        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=hparams.LEARNING_RATE_START,
            decay_steps=hparams.LEARNING_RATE_DECAY_STEPS,
            decay_rate=(hparams.LEARNING_RATE_END / hparams.LEARNING_RATE_START)**(1/(hparams.NUM_EPISODES // hparams.LEARNING_RATE_DECAY_STEPS +1)) # approx
        )
        
        self.q_network = QNetwork(state_size, action_size, hparams.HIDDEN_UNITS, self.lr_schedule, "q_main")
        self.target_network = QNetwork(state_size, action_size, hparams.HIDDEN_UNITS, self.lr_schedule, "q_target")
        self.update_target_network(tau=1.0) # Hard update initially

        self.replay_buffer = deque(maxlen=hparams.REPLAY_BUFFER_SIZE)
        
        self.epsilon = hparams.EPSILON_START
        # Calculate epsilon decay factor per step to reach EPSILON_MIN in EPSILON_DECAY_STEPS
        self.epsilon_decay_factor = (hparams.EPSILON_MIN / hparams.EPSILON_START)**(1/hparams.EPSILON_DECAY_STEPS)


    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.hparams.TAU
        if tau == 1.0: # Hard update
            self.target_network.model.set_weights(self.q_network.model.get_weights())
        else: # Soft update
            q_weights = self.q_network.model.get_weights()
            target_weights = self.target_network.model.get_weights()
            new_weights = [tau * q_w + (1 - tau) * t_w for q_w, t_w in zip(q_weights, target_weights)]
            self.target_network.model.set_weights(new_weights)

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def act(self, state, use_exploration=True):
        if use_exploration and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_reshaped = np.reshape(state, [1, self.state_size])
        q_values = self.q_network.predict(state_reshaped)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.replay_buffer) < self.hparams.BATCH_SIZE:
            return 0.0

        minibatch = random.sample(self.replay_buffer, self.hparams.BATCH_SIZE)
        
        states = np.array([experience[0] for experience in minibatch])
        action_indices = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch if experience[3] is not None]) # Handle None
        dones = np.array([experience[4] for experience in minibatch])

        # Get current Q-values for the states from the q_network
        current_q_values_batch = self.q_network.predict(states)
        
        # Get Q-values for next_states from target_network for Double DQN
        # For terminal states (gamma=0), next_q_values are not needed, but prepare for future gamma > 0
        next_q_values_target_net = np.zeros(self.hparams.BATCH_SIZE)
        if self.hparams.GAMMA > 0 and len(next_states) > 0: # Only if gamma > 0 and there are non-terminal next_states
            # Double DQN: Use q_network to select best action for next_state,
            # then use target_network to evaluate that action's Q-value.
            next_actions_q_net = np.argmax(self.q_network.predict(next_states), axis=1)
            next_q_values_from_target = self.target_network.predict(next_states)
            
            # Select the Q-value from target_network corresponding to the action chosen by q_network
            valid_next_state_indices = [i for i, exp in enumerate(minibatch) if exp[3] is not None]
            for i, original_idx in enumerate(valid_next_state_indices):
                if not dones[original_idx]: # Only for non-terminal states
                    next_q_values_target_net[original_idx] = next_q_values_from_target[i, next_actions_q_net[i]]

        targets_full_batch = np.copy(current_q_values_batch)

        for i in range(self.hparams.BATCH_SIZE):
            if dones[i]:
                targets_full_batch[i, action_indices[i]] = rewards[i]
            else: # This part is only relevant if gamma > 0 and not always done
                targets_full_batch[i, action_indices[i]] = rewards[i] + self.hparams.GAMMA * next_q_values_target_net[i]
        
        loss = self.q_network.update(states, targets_full_batch)
        
        if self.epsilon > self.hparams.EPSILON_MIN:
            self.epsilon *= self.epsilon_decay_factor # Decay epsilon per training step
            self.epsilon = max(self.hparams.EPSILON_MIN, self.epsilon)
        return loss

    def save(self, directory, episode):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.q_network.save(os.path.join(directory, f"q_network_ep{episode}.weights.h5"))
        self.target_network.save(os.path.join(directory, f"target_network_ep{episode}.weights.h5"))
        # Save HPARAMS and agent state (like epsilon)
        agent_state = {
            'episode': episode,
            'epsilon': self.epsilon,
            'hparams': vars(self.hparams) # Save a copy of hparams used for this training
        }
        with open(os.path.join(directory, f"agent_state_ep{episode}.json"), 'w') as f:
            json.dump(agent_state, f, indent=4)
        print(f"Saved model and agent state at episode {episode}")

    def load(self, directory, episode):
        self.q_network.load(os.path.join(directory, f"q_network_ep{episode}.weights.h5"))
        self.target_network.load(os.path.join(directory, f"target_network_ep{episode}.weights.h5"))
        with open(os.path.join(directory, f"agent_state_ep{episode}.json"), 'r') as f:
            agent_state = json.load(f)
        self.epsilon = agent_state['epsilon']
        # Potentially load and verify HPARAMS if needed, or use current HPARAMS
        print(f"Loaded model and agent state from episode {episode}, epsilon set to {self.epsilon:.4f}")


# --- Main Training Loop ---
if __name__ == "__main__":
    env = SurfaceCodeEnv(code_distance=HPARAMS.CODE_DISTANCE, error_prob=HPARAMS.ERROR_PROBABILITY)
    
    state_size = env.num_z_stabilizers
    action_size = 2**env.data_qubits
    
    agent = DQNAgent(state_size, action_size, HPARAMS)
    
    # --- Optional: Load existing model ---
    # LOAD_FROM_EPISODE = 10000 # Example
    # if LOAD_FROM_EPISODE > 0 and os.path.exists(os.path.join(HPARAMS.MODEL_SAVE_DIR, f"agent_state_ep{LOAD_FROM_EPISODE}.json")):
    #     print(f"Loading model from episode {LOAD_FROM_EPISODE}...")
    #     agent.load(HPARAMS.MODEL_SAVE_DIR, LOAD_FROM_EPISODE)
    #     start_episode = LOAD_FROM_EPISODE
    # else:
    #     start_episode = 0
    start_episode = 0 # Start fresh for this run

    rewards_history = []
    loss_history = []
    epsilon_history = []

    print(f"Starting training for {HPARAMS.NUM_EPISODES - start_episode} episodes...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Hyperparameters: {vars(HPARAMS)}")


    for episode in tqdm(range(start_episode, HPARAMS.NUM_EPISODES), desc="Training Progress"):
        state = env.reset()
        
        action_idx = agent.act(state)
        
        action_binary_str = format(action_idx, f'0{env.data_qubits}b')
        action_binary_vector = np.array([int(bit) for bit in action_binary_str])
        
        next_state, reward, done = env.step(action_binary_vector)
        
        agent.remember(state, action_idx, reward, next_state, done)
        
        current_loss = agent.replay()
        
        rewards_history.append(reward)
        loss_history.append(current_loss)
        epsilon_history.append(agent.epsilon)

        if (episode + 1) % HPARAMS.TARGET_NETWORK_UPDATE_FREQ == 0:
            agent.update_target_network()

        if (episode + 1) % HPARAMS.PRINT_EVERY_EPISODES == 0:
            avg_reward = np.mean(rewards_history[-HPARAMS.PRINT_EVERY_EPISODES:])
            avg_loss = np.mean([l for l in loss_history[-HPARAMS.PRINT_EVERY_EPISODES:] if l is not None and l > 0]) # Filter 0 loss
            current_lr = agent.q_network.model.optimizer.learning_rate
            if hasattr(current_lr, 'numpy'): # If it's a tf.Variable
                 current_lr = current_lr.numpy()
            tqdm.write(f"Ep {episode+1}/{HPARAMS.NUM_EPISODES} | Avg Reward (last {HPARAMS.PRINT_EVERY_EPISODES}): {avg_reward:.3f} | "
                       f"Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.4f} | LR: {current_lr:.6f}")
        
        if (episode + 1) % HPARAMS.SAVE_MODEL_EVERY_EPISODES == 0 and (episode+1) > 0:
            agent.save(HPARAMS.MODEL_SAVE_DIR, episode + 1)

    print("\n--- Training Finished ---")
    if not os.path.exists(HPARAMS.MODEL_SAVE_DIR):
        os.makedirs(HPARAMS.MODEL_SAVE_DIR)
    agent.save(HPARAMS.MODEL_SAVE_DIR, HPARAMS.NUM_EPISODES) # Save final model

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Smoothed Rewards
    window_size = HPARAMS.PRINT_EVERY_EPISODES
    if len(rewards_history) >= window_size:
        moving_avg_rewards = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        axs[0].plot(np.arange(window_size-1, len(rewards_history)), moving_avg_rewards, label=f'Moving Avg Reward (window {window_size})')
    axs[0].plot(rewards_history, alpha=0.3, label='Per-Episode Reward')
    axs[0].set_title('Training Rewards')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True)

    # Loss
    valid_losses = [l for l in loss_history if l is not None and l > 0]
    if valid_losses:
      axs[1].plot(valid_losses, label='Training Loss')
      if len(valid_losses) >= window_size:
          moving_avg_loss = np.convolve(valid_losses, np.ones(window_size)/window_size, mode='valid')
          axs[1].plot(np.arange(window_size-1, len(valid_losses)), moving_avg_loss, label=f'Moving Avg Loss (window {window_size})')
    axs[1].set_title('Training Loss (Huber)')
    axs[1].set_ylabel('Loss')
    axs[1].set_yscale('log') # Loss can vary a lot
    axs[1].legend()
    axs[1].grid(True)

    # Epsilon
    axs[2].plot(epsilon_history, label='Epsilon')
    axs[2].set_title('Epsilon Decay')
    axs[2].set_ylabel('Epsilon')
    axs[2].set_xlabel('Episode')
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(HPARAMS.MODEL_SAVE_DIR, "training_plots.png"))
    plt.show()

    # --- Testing Phase ---
    print("\n--- Testing Learned Policy (Greedy) ---")
    test_episodes = 500
    successful_corrections = 0
    if HPARAMS.NUM_EPISODES > 0 : # Only test if training happened
        # agent.load(HPARAMS.MODEL_SAVE_DIR, HPARAMS.NUM_EPISODES) # Load the final saved model
        print(f"Using agent from end of training (Episode {HPARAMS.NUM_EPISODES}).")
    
    for i in tqdm(range(test_episodes), desc="Testing Progress"):
        state = env.reset()
        original_errors = np.copy(env.current_x_errors)
        syndrome_str = "".join(map(str, state.astype(int)))

        action_idx = agent.act(state, use_exploration=False) # Greedy action
        
        action_binary_str = format(action_idx, f'0{env.data_qubits}b')
        action_binary_vector = np.array([int(bit) for bit in action_binary_str])
        
        _, reward, _ = env.step(action_binary_vector)
        
        if reward == 1.0:
            successful_corrections += 1
        
        if i < 5 : # Print details for first few test cases
            residual_errors = np.mod(original_errors + action_binary_vector, 2)
            print(f"\nTest Case {i+1}:")
            print(f"  Original X Errors: {original_errors} -> Syndrome: {syndrome_str}")
            print(f"  Agent Action Idx: {action_idx} -> Correction: {action_binary_vector}")
            print(f"  Residual X Errors: {residual_errors}")
            print(f"  Achieved Reward: {reward}")
            if reward == -1.0:
                 print(f"  Logical Error Detected: {env._has_logical_x_error(residual_errors)}")


    success_rate = (successful_corrections / test_episodes) * 100 if test_episodes > 0 else 0
    print(f"\nSuccess rate over {test_episodes} test episodes: {success_rate:.2f}%")