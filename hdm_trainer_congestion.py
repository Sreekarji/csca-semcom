"""
HDM Training with Curriculum Congestion
Cycles through tasks_per_csca = [1, 2, 4, 6, 10] during training.
Matches paper's Fig 9a methodology (2-20 tasks per CSCA).
"""
import sys, os, torch, numpy as np
sys.path.insert(0, r'D:\MP2\code\hdm')
sys.path.insert(0, r'D:\MP2\code\channel')
sys.path.insert(0, r'D:\MP2\code\evaluation')
sys.path.insert(0, r'D:\MP2\code\utils')
from reproducibility import set_seed
from datetime import datetime
set_seed(42)

from hdm_trainer import HDMTrainer

def ts(): return datetime.now().strftime('%H:%M:%S')

print(f'[{ts()}] HDM Training with Curriculum Congestion')
print(f'  tasks_schedule=[1, 2, 4, 6, 10]')

trainer = HDMTrainer(
    n_denoising_steps=6,
    tasks_schedule=[1, 2, 4, 6, 10],
)
print(f'  env: n_tasks={trainer.env.n_tasks}, difficulty={trainer.env.difficulty}')
print(f'  tasks_schedule={trainer.tasks_schedule}')

# Load from best checkpoint if exists
CKPT = r'D:\MP2\results\software\checkpoints'
best_ckpt = os.path.join(CKPT, 'hdm_medium_best.pt')
if os.path.exists(best_ckpt):
    ckpt = torch.load(best_ckpt, map_location=trainer.device, weights_only=False)
    trainer.han.load_state_dict(ckpt['han'])
    trainer.actor.load_state_dict(ckpt['actor'])
    trainer.critic.load_state_dict(ckpt['critic'])
    print(f'  Resumed from best checkpoint: ep={ckpt.get("episode","?")}')
else:
    print(f'  No checkpoint found, training from scratch')

best_isr = 0.0
isrs = []
for ep in range(1, 2001):
    r, cl, al, isr = trainer.train_batch_episode(batch_size=8)
    isrs.append(isr)
    tpc = trainer.tasks_schedule[trainer.current_episode % len(trainer.tasks_schedule)]

    if ep % 100 == 0:
        avg_isr = np.mean(isrs[-50:])
        print(f'[{ts()}] ep{ep}/2000 | tpc={tpc} | avg_ISR(50)={avg_isr:.3f} | actor={al:.4f}')
        ckpt_new = {
            'han': trainer.han.state_dict(),
            'actor': trainer.actor.state_dict(),
            'critic': trainer.critic.state_dict(),
            'episode': ep,
            'isr': avg_isr,
        }
        torch.save(ckpt_new, os.path.join(CKPT, f'hdm_congestion_ep{ep}.pt'))
        if avg_isr > best_isr:
            best_isr = avg_isr
            torch.save(ckpt_new, os.path.join(CKPT, 'hdm_congestion_best.pt'))
            print(f'  *** NEW BEST: {best_isr:.3f} at ep{ep} ***')
        if avg_isr >= 0.90:
            print(f'  90% TARGET REACHED at ep{ep}!')

print(f'\n=== HDM CONGESTION TRAINING DONE ===')
print(f'Best ISR: {best_isr:.3f}')
print(f'Final avg ISR (last 100 ep): {np.mean(isrs[-100:]):.3f}')
print(f'90% reached: {best_isr >= 0.90}')
