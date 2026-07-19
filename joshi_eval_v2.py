"""
joshi_eval_v2.py — Evaluate congestion-trained models across tasks_per_csca
Loads checkpoints from congestion training (hdm_congestion_best.pt, mlp_medium_best.pt, etc.)
Evaluates at tpc = [1, 2, 4, 6, 8, 10, 12, 15, 20] with 200 episodes each.
"""
import sys, os, csv, torch, numpy as np
sys.path.insert(0, r'D:\MP2\code\hdm')
sys.path.insert(0, r'D:\MP2\code\channel')
sys.path.insert(0, r'D:\MP2\code\evaluation')
sys.path.insert(0, r'D:\MP2\code\utils')
from reproducibility import set_seed
set_seed(42)

from han_network import HANNetwork
from ddpm_policy import HDMPolicy
from mlp_policy import MLPActor
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_isr
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = r'D:\MP2\results\software\checkpoints'

# ============================================================
# Load models
# ============================================================
def load_hdm(ckpt_name='hdm_congestion_best.pt'):
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
    hdm = HDMPolicy(action_dim=45, graph_emb_dim=256, n_denoising_steps=6).to(DEVICE)
    path = os.path.join(CKPT, ckpt_name)
    if not os.path.exists(path):
        path = os.path.join(CKPT, 'hdm_medium_best.pt')
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    han.load_state_dict(ckpt['han']); hdm.load_state_dict(ckpt['actor'])
    han.eval(); hdm.eval()
    print(f'HDM loaded: {os.path.basename(path)} (ep={ckpt.get("episode","?")})')
    return han, hdm

def load_mlp(ckpt_name='mlp_medium_best.pt'):
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
    actor = MLPActor(graph_emb_dim=256, task_emb_dim=256, action_dim=45,
                     hidden_dim=256, n_tasks=5).to(DEVICE)
    path = os.path.join(CKPT, ckpt_name)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    han.load_state_dict(ckpt['han']); actor.load_state_dict(ckpt['actor'])
    han.eval(); actor.eval()
    print(f'MLP loaded: {ckpt_name} (ep={ckpt.get("episode","?")})')
    return han, actor

def load_baseline(name, ckpt_name):
    archs = {'SAC': [256, 256, 45], 'PPO': [256, 45], 'AC': [128, 45]}
    arch = archs[name]
    layers = []
    in_d = 256
    for i, out_d in enumerate(arch):
        layers.append(nn.Linear(in_d, out_d))
        layers.append(nn.Sigmoid() if i == len(arch)-1 else nn.ReLU())
        in_d = out_d
    actor = nn.Sequential(*layers).to(DEVICE)
    path = os.path.join(CKPT, ckpt_name)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        actor.load_state_dict(ckpt['actor'])
        print(f'{name} loaded: {ckpt_name}')
    else:
        print(f'{name}: checkpoint not found, using random weights')
    actor.eval()
    return actor

# Load all models
han_hdm, hdm = load_hdm()
han_mlp, mlp = load_mlp()
# Baselines use HDM's HAN for encoding (paper-faithful: same state representation)
sac = load_baseline('SAC', 'sac_v2_best.pt')
ppo = load_baseline('PPO', 'ppo_v2_best.pt')
ac = load_baseline('AC', 'ac_v2_best.pt')

# ============================================================
# Evaluate across tasks_per_csca
# ============================================================
tpc_list = [1, 2, 4, 6, 8, 10, 12, 15, 20]
n_eval = 200

results = {name: [] for name in ['HDM', 'MLP', 'SAC', 'PPO', 'AC', 'Static']}

print(f'\n=== ISR vs TASKS (congestion-trained models, medium difficulty) ===')
print(f'{"tpc":>4} {"total":>6} {"HDM":>8} {"MLP":>8} {"SAC":>8} {"PPO":>8} {"AC":>8} {"Static":>8}')
print('-' * 68)

for tpc in tpc_list:
    # Fresh env for each tpc
    env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty='medium', tasks_per_csca=tpc)
    
    ep_results = {name: [] for name in results}
    
    for ep in range(n_eval):
        state = env.generate_state()
        
        # HDM
        g, _, m = han_hdm.encode_state(state)
        with torch.no_grad():
            a = hdm(g, message_embs=m)
        bw = a[:, :5]; relay = a[:, 5:30].reshape(1,5,5); mcs = a[:, 30:].reshape(1,5,3)
        r = env.step({'bandwidth': bw, 'relay': relay, 'mcs': mcs}, state)
        ep_results['HDM'].append(compute_isr(r['tasks']))
        
        # MLP
        g2, _, m2 = han_mlp.encode_state(state)
        with torch.no_grad():
            a2 = mlp(g2, message_embs=m2)
        bw2 = a2[:, :5]; relay2 = a2[:, 5:30].reshape(1,5,5); mcs2 = a2[:, 30:].reshape(1,5,3)
        r2 = env.step({'bandwidth': bw2, 'relay': relay2, 'mcs': mcs2}, state)
        ep_results['MLP'].append(compute_isr(r2['tasks']))
        
        # SAC, PPO, AC (use HDM's HAN for encoding)
        for name, model in [('SAC', sac), ('PPO', ppo), ('AC', ac)]:
            with torch.no_grad():
                a_b = model(g)
            bw_b = a_b[:, :5]; relay_b = a_b[:, 5:30].reshape(1,5,5); mcs_b = a_b[:, 30:].reshape(1,5,3)
            r_b = env.step({'bandwidth': bw_b, 'relay': relay_b, 'mcs': mcs_b}, state)
            ep_results[name].append(compute_isr(r_b['tasks']))
        
        # Static
        sa = torch.ones(1, 45, device=DEVICE) * 0.5
        rs = env.step({'bandwidth': sa[:, :5], 'relay': sa[:, 5:30].reshape(1,5,5), 'mcs': sa[:, 30:].reshape(1,5,3)}, state)
        ep_results['Static'].append(compute_isr(rs['tasks']))
    
    # Average over episodes
    for name in results:
        results[name].append(np.mean(ep_results[name]))
    
    print(f'{tpc:>4} {tpc*5:>6} {results["HDM"][-1]:>8.3f} {results["MLP"][-1]:>8.3f} '
          f'{results["SAC"][-1]:>8.3f} {results["PPO"][-1]:>8.3f} {results["AC"][-1]:>8.3f} '
          f'{results["Static"][-1]:>8.3f}')

# ============================================================
# Key comparison points
# ============================================================
print(f'\n=== KEY COMPARISON POINTS ===')
for tpc_target in [10, 20]:
    idx = tpc_list.index(tpc_target)
    print(f'\ntpccsca={tpc_target} (total={tpc_target*5} tasks):')
    for name in ['HDM', 'MLP', 'SAC', 'PPO', 'AC', 'Static']:
        isr = results[name][idx]
        vs_static = (isr - results['Static'][idx]) / max(results['Static'][idx], 1e-6) * 100
        print(f'  {name:<8} ISR={isr:.3f} (vs Static: {vs_static:+.1f}%)')

print(f'\n=== BEST ISR PER METHOD ===')
for name in results:
    best = max(results[name])
    best_tpc = tpc_list[results[name].index(best)]
    print(f'{name:<8} best_ISR={best:.3f} at tpc={best_tpc}')

# ============================================================
# Save CSV
# ============================================================
csv_path = r'D:\Desktop\joshi_eval_v2_isr_vs_tasks.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['tpc', 'total_tasks', 'HDM', 'MLP', 'SAC', 'PPO', 'AC', 'Static'])
    for i, tpc in enumerate(tpc_list):
        w.writerow([tpc, tpc*5, results['HDM'][i], results['MLP'][i],
                     results['SAC'][i], results['PPO'][i], results['AC'][i], results['Static'][i]])
print(f'\nSaved: {csv_path}')
