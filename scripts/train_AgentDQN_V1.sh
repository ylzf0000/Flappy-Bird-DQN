python AgentDQN_V1.py \
  --mode=train \
  --experience_name="DQN" \
  --model_class=DQN \
  --learning_rate=3e-4 \
  --weight_decay=1e-5 \
  --train_steps=3000000 \
  --replay_memory_size=102400 \
  --save_model_steps=10000 \
  --decision_interval=4 \
  --decision_interval_percent=100000

