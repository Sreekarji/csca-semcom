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
from cscqi import compute_isr, compute_cscqi

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = r'D:\MP2\results\software\checkpoints'

def load_mlp():
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
    actor = MLPActor(graph_emb_dim=256, task_emb_dim=256, action_dim=45,
                     hidden_dim=256, n_tasks=5).to(DEVICE)
    ckpt = torch.load(os.path.join(CKPT, 'mlp_medium_best.pt'), map_location=DEVICE, weights_only=False)
    han.load_state_dict(ckpt['han']); actor.load_state_dict(ckpt['actor'])
    han.eval(); actor.eval()
    return han, actor

def load_hdm():
    han = HANNetwork(hidden_channels=256, num_heads=8, num_layers=3,
                     n_cscas=5, n_relays=5, n_messages=5, n_base_stations=5).to(DEVICE)
    hdm = HDMPolicy(action_dim=45, graph_emb_dim=256, n_denoising_steps=6).to(DEVICE)
    ckpt = torch.load(os.path.join(CKPT, 'hdm_medium_best.pt'), map_location=DEVICE, weights_only=False)
    han.load_state_dict(ckpt['han']); hdm.load_state_dict(ckpt['actor'])
    han.eval(); hdm.eval()
    return han, hdm

han_mlp, mlp = load_mlp()
han_hdm, hdm = load_hdm()

tpc_list = [1, 2, 4, 6, 8]
n_eval = 300  # episodes per point

results = {'HDM': [], 'MLP': [], 'Static': []}

print('tasks_per_csca | total_tasks | HDM ISR | MLP ISR | Static ISR')
print('-' * 65)

for tpc in tpc_list:
    env = MultiCSCAEnvironment(n_cscas=5, n_relays=5, difficulty='medium', tasks_per_csca=tpc)
    row = {'HDM': [], 'MLP': [], 'Static': []}
    
    for ep in range(n_eval):
        state = env.generate_state()
        
        # HDM
        g, _, m = han_hdm.encode_state(state)
        with torch.no_grad():
            a = hdm(g, message_embs=m)
        bw = a[:, :5]; relay = a[:, 5:30].reshape(1,5,5); mcs = a[:, 30:].reshape(1,5,3)
        r = env.step({'bandwidth': bw, 'relay': relay, 'mcs': mcs}, state)
        row['HDM'].append(compute_isr(r['tasks']))
        
        # MLP
        g2, _, m2 = han_mlp.encode_state(state)
        with torch.no_grad():
            a2 = mlp(g2, message_embs=m2)
        bw2 = a2[:, :5]; relay2 = a2[:, 5:30].reshape(1,5,5); mcs2 = a2[:, 30:].reshape(1,5,3)
        r2 = env.step({'bandwidth': bw2, 'relay': relay2, 'mcs': mcs2}, state)
        row['MLP'].append(compute_isr(r2['tasks']))
        
        # Static
        s_act = torch.ones(1, 45, device=DEVICE) * 0.5
        rs = env.step({'bandwidth': s_act[:, :5], 'relay': s_act[:, 5:30].reshape(1,5,5),
                       'mcs': s_act[:, 30:].reshape(1,5,3)}, state)
        row['Static'].append(compute_isr(rs['tasks']))
    
    for m in results:
        results[m].append(np.mean(row[m]))
    
    print(f'tpc={tpc:2d} (total={tpc*5:3d}) | HDM={results["HDM"][-1]:.3f} | MLP={results["MLP"][-1]:.3f} | Static={results["Static"][-1]:.3f}')

print('\n=== BEST ISR ACHIEVED ===')
print(f'HDM max: {max(results["HDM"]):.3f}')
print(f'MLP max: {max(results["MLP"]):.3f}')
print(f'Static max: {max(results["Static"]):.3f}')
hdm_vs_static = (max(results['HDM']) - max(results['Static'])) / max(results['Static']) * 100
print(f'HDM vs Static improvement: {hdm_vs_static:+.1f}%')
print(f'90% target reached: HDM={max(results["HDM"]) >= 0.90}, MLP={max(results["MLP"]) >= 0.90}')

# Save CSV
with open(r'D:\Desktop\final_eval_medium.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['tpc', 'total_tasks', 'HDM', 'MLP', 'Static'])
    for i, tpc in enumerate(tpc_list):
        w.writerow([tpc, tpc*5, results['HDM'][i], results['MLP'][i], results['Static'][i]])
print('Saved: D:\\Desktop\\final_eval_medium.csv')
