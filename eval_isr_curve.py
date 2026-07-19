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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = r'D:\MP2\results\software\checkpoints'

def load_best_mlp():
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
    actor = MLPActor(graph_emb_dim=256, task_emb_dim=256, action_dim=45,
                     hidden_dim=256, n_tasks=5).to(DEVICE)
    ckpt = torch.load(os.path.join(CKPT, 'mlp_medium_best.pt'), map_location=DEVICE, weights_only=False)
    han.load_state_dict(ckpt['han']); actor.load_state_dict(ckpt['actor'])
    han.eval(); actor.eval()
    print(f'MLP loaded from ep={ckpt.get("episode","?")} ISR={ckpt.get("isr","?"):.3f}')
    return han, actor

def load_best_hdm():
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
    hdm = HDMPolicy(action_dim=45, graph_emb_dim=256, n_denoising_steps=6).to(DEVICE)
    ckpt = torch.load(os.path.join(CKPT, 'hdm_medium_best.pt'), map_location=DEVICE, weights_only=False)
    han.load_state_dict(ckpt['han']); hdm.load_state_dict(ckpt['actor'])
    han.eval(); hdm.eval()
    print(f'HDM loaded from ep={ckpt.get("episode","?")}')
    return han, hdm

han_mlp, mlp = load_best_mlp()
han_hdm, hdm = load_best_hdm()

# Evaluate at tpc=1 (training distribution) and increasingly tight constraints
n_eval = 500  # enough for stable ISR estimate

# Test 1: tpc=1 at the training distribution (medium)
print('\n=== EVALUATION AT TRAINING DISTRIBUTION (tpc=1, medium) ===')
env_med = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty='medium', tasks_per_csca=1)

for name, han_m, model in [('HDM', han_hdm, hdm), ('MLP', han_mlp, mlp)]:
    isrs = []
    for _ in range(n_eval):
        state = env_med.generate_state()
        g, _, m = han_m.encode_state(state)
        with torch.no_grad():
            a = model(g, message_embs=m)
        bw = a[:, :5]; relay = a[:, 5:30].reshape(1,5,5); mcs = a[:, 30:].reshape(1,5,3)
        r = env_med.step({'bandwidth': bw, 'relay': relay, 'mcs': mcs}, state)
        isrs.append(compute_isr(r['tasks']))
    print(f'{name}: ISR={np.mean(isrs):.3f} +/- {np.std(isrs):.3f}')

# Static baseline
static_isrs = []
for _ in range(n_eval):
    state = env_med.generate_state()
    sa = torch.ones(1, 45, device=DEVICE) * 0.5
    rs = env_med.step({'bandwidth': sa[:,:5], 'relay': sa[:,5:30].reshape(1,5,5), 'mcs': sa[:,30:].reshape(1,5,3)}, state)
    static_isrs.append(compute_isr(rs['tasks']))
print(f'Static: ISR={np.mean(static_isrs):.3f} +/- {np.std(static_isrs):.3f}')

# Test 2: ISR degradation as data size increases (simulates increasing load, tpc=1)
print('\n=== ISR vs LOAD (simulated by data size increase, tpc=1 medium) ===')
data_size_scales = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]  # multiplier on medium data size
print('Scale | Total_data_equiv | HDM   | MLP   | Static')

results = {'HDM': [], 'MLP': [], 'Static': []}
for scale in data_size_scales:
    env_scaled = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty='medium', tasks_per_csca=1)
    
    hdm_isrs, mlp_isrs, stat_isrs = [], [], []
    for _ in range(300):
        state = env_scaled.generate_state()
        # Scale up data sizes to simulate higher load
        for i in range(5):
            state['SCt']['data_sizes'][i] = min(state['SCt']['data_sizes'][i] * scale, 5e6)
        
        g_h, _, m_h = han_hdm.encode_state(state)
        with torch.no_grad():
            a_h = hdm(g_h, message_embs=m_h)
        bw_h = a_h[:,:5]; relay_h = a_h[:,5:30].reshape(1,5,5); mcs_h = a_h[:,30:].reshape(1,5,3)
        r_h = env_scaled.step({'bandwidth': bw_h, 'relay': relay_h, 'mcs': mcs_h}, state)
        hdm_isrs.append(compute_isr(r_h['tasks']))
        
        g_m, _, m_m = han_mlp.encode_state(state)
        with torch.no_grad():
            a_m = mlp(g_m, message_embs=m_m)
        bw_m = a_m[:,:5]; relay_m = a_m[:,5:30].reshape(1,5,5); mcs_m = a_m[:,30:].reshape(1,5,3)
        r_m = env_scaled.step({'bandwidth': bw_m, 'relay': relay_m, 'mcs': mcs_m}, state)
        mlp_isrs.append(compute_isr(r_m['tasks']))
        
        sa = torch.ones(1, 45, device=DEVICE) * 0.5
        rs = env_scaled.step({'bandwidth': sa[:,:5], 'relay': sa[:,5:30].reshape(1,5,5), 'mcs': sa[:,30:].reshape(1,5,3)}, state)
        stat_isrs.append(compute_isr(rs['tasks']))
    
    h, m_val, s = np.mean(hdm_isrs), np.mean(mlp_isrs), np.mean(stat_isrs)
    results['HDM'].append(h); results['MLP'].append(m_val); results['Static'].append(s)
    equiv_tasks = int(scale * 2)
    print(f'{scale:.1f}x  | ~{equiv_tasks} tasks equiv    | {h:.3f} | {m_val:.3f} | {s:.3f}')

print(f'\nPeak ISR: HDM={max(results["HDM"]):.3f}, MLP={max(results["MLP"]):.3f}, Static={max(results["Static"]):.3f}')
print(f'90% reached: HDM={max(results["HDM"])>=0.90}, MLP={max(results["MLP"])>=0.90}')
print(f'HDM vs Static at 0.5x load: {(results["HDM"][0]-results["Static"][0])/results["Static"][0]*100:+.1f}%')
print(f'HDM vs Static at 1.0x load: {(results["HDM"][1]-results["Static"][1])/results["Static"][1]*100:+.1f}%')

with open(r'D:\Desktop\isr_curve_v2.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['load_scale', 'equiv_tasks', 'HDM', 'MLP', 'Static'])
    for i, scale in enumerate(data_size_scales):
        w.writerow([scale, int(scale*2), results['HDM'][i], results['MLP'][i], results['Static'][i]])
print('Saved: D:\\Desktop\\isr_curve_v2.csv')
