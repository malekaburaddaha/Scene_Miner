# Default composite loss weights for SceneMinerLoss.
#
#   w_embed  : InfoNCE contrastive loss on scene embeddings
#   w_tag    : multi-label BCE loss for scene-tag classification
#   w_risk   : binary cross-entropy loss for risk score head
#   w_caption: cross-entropy on VLM caption tokens (active in stage 3)

loss = dict(
    w_embed=1.0,
    w_tag=0.5,
    w_risk=0.2,
    w_caption=0.3,
    temperature=0.07,
)
