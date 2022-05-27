import torch


class Perturber:
    def __init__(self, std_scale=2e-2):
        self.std_scale = std_scale
        self.max_state = torch.tensor([2.4, 2.4, 0.418, 1.5])

    def perturb(self, state):
        """
        State has (x, v, theta, omega)
        These values can't exceed (2.4, 2.4, 0.418, 1.5)
        as seen from the rollouts

        For each value, we look at distance to boundary
        for x=2.3, distance to boundary is 0.1
        perturbation for x ~ N(0, 0.1 * self.std_scale)

        for omega=-1.25, distance to boundary is 0.25
        perturbation for omega ~ N(0, 0.2 * self.std_scale)
        and so on

        position_saliency = (model_out - model_out_for_perturbed_position) / perturbation_for_position
        omega_saliency = (model_out - model_out_for_perturbed_omega) / perturbation_for_omega
        and so on
        """
        if state.shape != self.max_state.shape:
            max_state = self.max_state[1:]
        else:
            max_state = self.max_state

        dist_to_left = torch.abs(state + max_state)
        dist_to_right = torch.abs(max_state - state)
        min_distance = torch.vstack([dist_to_left, dist_to_right]).min(0).values
        std = min_distance * self.std_scale
        pert = torch.distributions.normal.Normal(torch.zeros_like(state), std).sample()
        left_approach = state.unsqueeze(0) - torch.diag_embed(pert)
        right_approach = state.unsqueeze(0) + torch.diag_embed(pert)
        return left_approach, right_approach, pert
