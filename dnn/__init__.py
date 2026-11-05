"""diffusionNN: a text-to-network diffusion model.

A diffusion model with a text2image-style architecture (UNet + text
cross-attention + classifier-free guidance) where the "image" being
denoised is the weight tensor of a small, fixed-size neural network.
Training pairs are (network, caption): tiny MLP controllers for games
and animation tasks, trained on synthetic environments, each described
in English. Sampling from a prompt yields a brand-new network that is
executed directly in the target environment.
"""

__version__ = "0.1.0"
