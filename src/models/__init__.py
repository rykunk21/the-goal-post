"""ML Models package.

Contains VAE, Transition NN, and related components.
"""

# Import VAE + InfoNCE modules
from .vae_infonce import (
    VAEEncoder,
    VAEDecoder,
    VAE,
    InfoNCELoss,
    VAEWithInfoNCE,
    create_vae_trainer,
    train_epoch,
    evaluate,
    get_team_embedding
)

# Transition Probability NN module
# TODO: Implement TransitionProbabilityNN
class TransitionProbabilityNN:
    pass


# Training components
# TODO: Implement VAEFeedbackTrainer
class VAEFeedbackTrainer:
    pass
