# Quantum Error Correction with Reinforcement Learning

This project implements a deep reinforcement learning approach to quantum error correction using surface codes. It uses a Deep Q-Network (DQN) to learn optimal error correction strategies for quantum computation.

## Overview

The system trains an agent to detect and correct errors in a surface code quantum memory using advanced deep learning techniques. The implementation features an enhanced DQN architecture with several modern improvements for stable and efficient learning.

### Key Features

- Surface code quantum error correction simulation
- Enhanced Deep Q-Network (DQN) implementation with:
  - Double DQN architecture
  - Dueling networks
  - Experience replay
  - Batch normalization
  - Convolutional layers for syndrome pattern recognition
  - Adaptive learning rate scheduling
  - Epsilon-greedy exploration with decay
  - Syndrome history tracking
  
## Technical Details

### Hyperparameters
- Code Distance: 3 (configurable)
- Error Probability: 0.05
- Measurement Error Probability: 0.01
- Training Episodes: 30,000
- Max Correction Rounds per Episode: 3
- Replay Buffer Size: 50,000
- Batch Size: 64
- Discount Factor (γ): 0.95
- Learning Rate: 5e-5 to 1e-5 (exponential decay)

### Architecture
- Input: Syndrome measurements (with optional history)
- Neural Network: 
  - Convolutional layers for syndrome pattern processing
  - Dense layers [256, 128] with ReLU activation
  - Dueling architecture for value/advantage separation
  - Batch normalization and dropout (0.1) for regularization

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- tqdm

## Usage

1. Train a new agent:
```python
python first-agent.py
```

This will:
- Initialize the surface code environment
- Train the DQN agent for 30,000 episodes
- Save model checkpoints every 2,500 episodes
- Generate training visualization plots
- Run final evaluation tests

## Results and Visualization

The training process generates several metrics that are plotted and saved:
- Episode rewards
- Training loss
- Epsilon decay
- Logical error rate
- Average rounds per episode
- Learning rate decay

Results are saved in the `super_enhanced_dqn_qec_model` directory:
- Model weights: `q_network_epXXXX.weights.h5`
- Agent state: `agent_state_epXXXX.json`
- Training plots: `training_summary_dX.png`

## Performance

The trained agent demonstrates:
- Successful error correction strategies
- Low logical error rates
- Efficient correction (minimal rounds)
- Stable learning behavior

The system is tested with 500 episodes post-training to validate performance and generalization.

## Implementation Details

### Key Components

1. `EnhancedSurfaceCodeEnv`: Quantum surface code simulation environment
2. `EnhancedDQNAgent`: Main reinforcement learning agent
3. `EnhancedQNetwork`: Neural network architecture
4. `SurfaceCodeGenerator`: Surface code structure generator
5. `Hyperparameters`: Configurable training parameters

### Training Process

The agent learns through episodes where it:
1. Observes syndrome measurements
2. Chooses correction operations
3. Receives rewards based on correction success
4. Updates its policy through experience replay
5. Adapts exploration rate and learning parameters

## License

MIT License

## Acknowledgments

This implementation builds on fundamental concepts from:
- Surface code quantum error correction
- Deep Q-Learning with modern improvements
- TensorFlow and Keras frameworks
