import torch
import torch.nn.functional as F

class PPOService:
    def __init__(self):
        self.clip_eps = 0.2

    def run(self, batch, policy, value, optimizer, profile):
        logits = policy(batch["state"], profile)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(batch["action"])
        ratio = torch.exp(logp - batch["logp"])
        v = value.run(batch["state"], profile)
        advantage = batch["reward"] - v.detach()
        policy_loss = -torch.mean(
            torch.min(
                ratio * advantage,
                torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps
                ) * advantage
            )
        )
        value_loss = F.mse_loss(v, batch["reward"])
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"policy_loss": policy_loss.item(),"value_loss": value_loss.item(),}
