"""
Baseline Trainer v2 — SAC, PPO, AC with congestion curriculum
Uses same HAN as HDM (loaded from checkpoint) for fair comparison.
Cycles through tasks_per_csca = [1, 2, 4, 6, 10] like HDM training.
"""
import sys, os, torch, numpy as np
sys.path.insert(0, r'D:\MP2\code\hdm')
sys.path.insert(0, r'D:\MP2\code\channel')
sys.path.insert(0, r'D:\MP2\code\evaluation')
sys.path.insert(0, r'D:\MP2\code\experiments')
sys.path.insert(0, r'D:\MP2\code\utils')
from reproducibility import set_seed
from datetime import datetime
set_seed(42)

import torch.nn as nn
from han_network import HANNetwork
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_cscqi, compute_isr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = r'D:\MP2\results\software\checkpoints'

def ts(): return datetime.now().strftime('%H:%M:%S')

# ============================================================
# Load shared HAN from HDM checkpoint (paper-faithful: same state encoding)
# ============================================================
han = HANNetwork(
    hidden_channels=256, num_heads=8, num_layers=3,
    n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5,
).to(DEVICE)

hdm_ckpt_path = os.path.join(CKPT, 'hdm_medium_best.pt')
if os.path.exists(hdm_ckpt_path):
    ckpt = torch.load(hdm_ckpt_path, map_location=DEVICE, weights_only=False)
    han.load_state_dict(ckpt['han'])
    print(f'[{ts()}] Loaded shared HAN from hdm_medium_best.pt (ep={ckpt.get("episode","?")})')
else:
    print(f'[{ts()}] WARNING: No HDM checkpoint found, using random HAN weights')
han.eval()

# ============================================================
# Baseline actors (matching train_baselines.py architecture)
# ============================================================
def make_sac_actor():
    return nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 45), nn.Sigmoid()
    ).to(DEVICE)

def make_ppo_actor():
    return nn.Sequential(
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 45), nn.Sigmoid()
    ).to(DEVICE)

def make_ac_actor():
    return nn.Sequential(
        nn.Linear(256, 128), nn.Tanh(),
        nn.Linear(128, 45), nn.Sigmoid()
    ).to(DEVICE)

def make_critic():
    return nn.Sequential(
        nn.Linear(256 + 45, 256), nn.ReLU(),
        nn.Linear(256, 1)
    ).to(DEVICE)

def parse_action(action, n_cscas=5, n_relays=5, n_mcs=3):
    bw = action[:, :n_cscas]
    relay = action[:, n_cscas:n_cscas + n_cscas * n_relays].reshape(1, n_cscas, n_relays)
    mcs = action[:, n_cscas + n_cscas * n_relays:].reshape(1, n_cscas, n_mcs)
    return {'bandwidth': bw, 'relay': relay, 'mcs': mcs}

def generate_intent_vectors(n_tasks):
    intents = []
    for i in range(n_tasks):
        if i % 3 == 0:
            intents.append([np.random.uniform(0.7, 1.0), np.random.uniform(0.3, 0.6)])
        elif i % 3 == 1:
            intents.append([np.random.uniform(0.1, 0.4), np.random.uniform(0.7, 1.0)])
        else:
            intents.append([np.random.uniform(0.4, 0.7), np.random.uniform(0.4, 0.7)])
    return intents

# ============================================================
# Training loop
# ============================================================
TASKS_SCHEDULE = [1, 2, 4, 6, 10]

def train_baseline(name, make_actor_fn, n_episodes=2000):
    print(f'\n[{ts()}] Training {name} for {n_episodes} episodes')
    
    actor = make_actor_fn()
    critic = make_critic()
    opt_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    
    isrs = []
    best_isr = 0.0
    
    for ep in range(1, n_episodes + 1):
        tpc = TASKS_SCHEDULE[(ep - 1) % len(TASKS_SCHEDULE)]
        env = MultiCSCAEnvironment(
            n_cscas=5, n_relays=5, bandwidth_total_hz=5e6,
            difficulty='medium', tasks_per_csca=tpc,
        )
        
        state = env.generate_state()
        intent_vectors = generate_intent_vectors(env.n_tasks)
        state['SCt']['delay_intents'] = [max(0.1, (1.0 - iv[0]) * 5.0) for iv in intent_vectors]
        state['SCt']['quality_intents'] = [iv[1] for iv in intent_vectors]
        
        with torch.no_grad():
            graph_emb, _, _ = han.encode_state(state, intent_vectors=intent_vectors)
        
        action = actor(graph_emb)
        parsed = parse_action(action)
        result = env.step(parsed, state)
        tasks = result['tasks']
        
        # CSCQI reward with hard negative
        cscqi_vals = [compute_cscqi(t['tau_S'], t['vartheta_S'], t['tau_S_int'], t['vartheta_S_int']) for t in tasks]
        reward = float(np.clip(np.mean(cscqi_vals), -5.0, 5.0))
        isr = compute_isr(tasks)
        isrs.append(isr)
        
        # Critic update (DDPG style)
        with torch.no_grad():
            next_action = actor(graph_emb)
            next_q = critic(torch.cat([graph_emb, next_action], dim=-1))
            target = reward + 0.95 * next_q
        
        q_val = critic(torch.cat([graph_emb.detach(), action.detach()], dim=-1))
        critic_loss = nn.MSELoss()(q_val, target)
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
        
        # Actor update (DDPG style)
        new_action = actor(graph_emb)
        q_new = critic(torch.cat([graph_emb.detach(), new_action], dim=-1))
        actor_loss = -q_new.mean()
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()
        
        # Logging and checkpoints
        if ep % 200 == 0:
            avg_isr = np.mean(isrs[-100:])
            print(f'[{ts()}] {name} ep{ep}/{n_episodes} | tpc={tpc} | avg_ISR(100)={avg_isr:.3f} | reward={reward:.4f}')
            ckpt = {'actor': actor.state_dict(), 'critic': critic.state_dict(), 'episode': ep, 'isr': avg_isr}
            torch.save(ckpt, os.path.join(CKPT, f'{name.lower()}_v2_ep{ep}.pt'))
            if avg_isr > best_isr:
                best_isr = avg_isr
                torch.save(ckpt, os.path.join(CKPT, f'{name.lower()}_v2_best.pt'))
                print(f'  *** NEW BEST: {best_isr:.3f} at ep{ep} ***')
    
    final_isr = np.mean(isrs[-100:])
    print(f'\n[{ts()}] {name} DONE | Final avg ISR (last 100): {final_isr:.3f} | Best ISR: {best_isr:.3f}')
    return final_isr, best_isr

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print(f'[{ts()}] Baseline Training v2 — Congestion Curriculum')
    print(f'  tasks_schedule={TASKS_SCHEDULE}')
    print(f'  difficulty=medium, n_episodes=2000 each')
    
    results = {}
    for name, make_fn in [('SAC', make_sac_actor), ('PPO', make_ppo_actor), ('AC', make_ac_actor)]:
        final, best = train_baseline(name, make_fn, n_episodes=2000)
        results[name] = (final, best)
    
    print(f'\n{"="*50}')
    print(f'BASELINE TRAINING COMPLETE')
    print(f'{"="*50}')
    print(f'{"Method":<8} {"Final ISR":>10} {"Best ISR":>10}')
    print(f'{"-"*30}')
    for name, (final, best) in results.items():
        print(f'{name:<8} {final:>10.3f} {best:>10.3f}')
