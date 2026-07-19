"""
HDM Continue v2 — Resume from hdm_best.pt with congestion curriculum
Runs 3000 more episodes with tasks_schedule=[1,2,4,6,10]
Uses actor loss scaling fix (advantage * 10.0)
"""
import sys, os, torch, numpy as np, csv
sys.path.insert(0, r'D:\MP2\code\hdm')
sys.path.insert(0, r'D:\MP2\code\channel')
sys.path.insert(0, r'D:\MP2\code\evaluation')
sys.path.insert(0, r'D:\MP2\code\utils')
from reproducibility import set_seed
from datetime import datetime
set_seed(42)

from hdm_trainer import HDMTrainer

def ts(): return datetime.now().strftime('%H:%M:%S')

CKPT = r'D:\MP2\results\software\checkpoints'

# Initialize with congestion curriculum
trainer = HDMTrainer(
    n_denoising_steps=6,
    tasks_schedule=[1, 2, 4, 6, 10],
)
print(f'[{ts()}] HDM Continue v2 — Congestion Curriculum')
print(f'  tasks_schedule={trainer.tasks_schedule}')
print(f'  env: n_tasks={trainer.env.n_tasks}, difficulty={trainer.env.difficulty}')

# Load from best checkpoint
best_ckpt_path = os.path.join(CKPT, 'hdm_medium_best.pt')
if not os.path.exists(best_ckpt_path):
    best_ckpt_path = os.path.join(CKPT, 'hdm_congestion_best.pt')

ckpt = torch.load(best_ckpt_path, map_location=trainer.device, weights_only=False)
trainer.han.load_state_dict(ckpt['han'])
trainer.actor.load_state_dict(ckpt['actor'])
trainer.critic.load_state_dict(ckpt['critic'])
start_ep = ckpt.get('episode', 0)
print(f'  Resuming from: {os.path.basename(best_ckpt_path)} ep={start_ep}')

best_isr = ckpt.get('isr', 0.0)
print(f'  Previous best ISR: {best_isr:.3f}')

isrs = []
for ep in range(1, 3001):
    r, cl, al, isr = trainer.train_batch_episode(batch_size=8)
    isrs.append(isr)
    total_ep = start_ep + ep
    tpc = trainer.tasks_schedule[trainer.current_episode % len(trainer.tasks_schedule)]

    if ep % 100 == 0:
        avg_isr = np.mean(isrs[-50:])
        lp = getattr(trainer, '_last_log_prob', 0.0)
        adv = getattr(trainer, '_last_advantage', 0.0)
        print(f'[{ts()}] ep{total_ep} (+{ep}) | tpc={tpc} | avg_ISR(50)={avg_isr:.3f} | '
              f'actor={al:.4f} | logP={lp:.4f} | adv={adv:.4f}')
        
        ckpt_new = {
            'han': trainer.han.state_dict(),
            'actor': trainer.actor.state_dict(),
            'critic': trainer.critic.state_dict(),
            'episode': total_ep,
            'isr': avg_isr,
        }
        torch.save(ckpt_new, os.path.join(CKPT, f'hdm_congestion_ep{total_ep}.pt'))
        
        if avg_isr > best_isr:
            best_isr = avg_isr
            torch.save(ckpt_new, os.path.join(CKPT, 'hdm_congestion_best.pt'))
            print(f'  *** NEW BEST: {best_isr:.3f} at ep{total_ep} ***')
        if avg_isr >= 0.90:
            print(f'  90% TARGET REACHED at ep{total_ep}!')

print(f'\n=== HDM CONTINUE v2 DONE ===')
print(f'Best ISR: {best_isr:.3f}')
print(f'Final avg ISR (last 100 ep): {np.mean(isrs[-100:]):.3f}')
print(f'90% reached: {best_isr >= 0.90}')

with open(r'D:\Desktop\hdm_continue_v2_log.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ep_relative', 'ep_total', 'isr'])
    for i, isr_val in enumerate(isrs):
        w.writerow([i+1, start_ep+i+1, isr_val])
