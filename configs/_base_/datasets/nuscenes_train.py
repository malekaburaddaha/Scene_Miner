# Base dataset config — nuScenes v1.0-trainval, training split
# 6 cameras (no portrait cam), 2 Hz annotation rate.

data = dict(
    type="NuScenesDataset",
    dataroot="/data/nuscenes",
    version="v1.0-trainval",
    split="train",
    val_split="val",            # used by OccEvalHook to build the val loader
    num_cams=6,
    image_size=(448, 800),
    verbose=False,
)
