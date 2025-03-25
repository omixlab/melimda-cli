from deepchem.feat import RdkitGridFeaturizer

grid_featurizer = RdkitGridFeaturizer(
    voxel_width=1.0,
    box_width=20,
    feature_types=["ecfp", "splif", "hbond", "salt_bridge", "pi_stack", "cation_pi", "shape"],
    ecfp_power=9,
    splif_power=9,
    flatten=True
)

features = grid_featurizer.featurize([('ligand.sdf', 'receptor.sdf')])
print(features)
