from hestia.similarity import calculate_similarity, sim_df2mtx
import numpy as np
import pandas as pd


def test_fingerprint_alignment():
    smiles = [
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([Br])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H]c1c(c(c(c(c1[H])Cl)[H])N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])N=C(N3[H])[H])[H]'
    ]
    df = pd.DataFrame({'smiles': smiles})
    sim_df = calculate_similarity(
        df, data_type='small_molecule',
        similarity_metric='fingerprint',
        field_name='smiles'
    )
    objective_df = pd.DataFrame({
        'query': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'target': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'metric': [1.0, 0.781818, 0.793103, 0.781818, 1.000000, 0.781818,
                   0.793103, 0.781818, 1.000000]
    })
    assert sim_df['query'].tolist() == objective_df['query'].tolist()
    assert sim_df['target'].tolist() == objective_df['target'].tolist()
    np.testing.assert_allclose(sim_df['metric'].tolist(),
                               objective_df['metric'].tolist(), rtol=0.001)

    sim_df2 = calculate_similarity(
        df, data_type='small_molecule',
        similarity_metric='fingerprint',
        field_name='smiles', threshold=0.8
    )

    objective_df2 = pd.DataFrame({
        'query': [0, 1, 2],
        'target': [0, 1, 2],
        'metric': [1.0, 1.0, 1.0]
    })
    assert sim_df2['query'].tolist() == objective_df2['query'].tolist()
    assert sim_df2['target'].tolist() == objective_df2['target'].tolist()
    np.testing.assert_allclose(sim_df2['metric'].to_numpy(),
                               objective_df2['metric'].to_numpy())


def test_simdf2mtx():
    objective = np.array(
          [[1., 0.7817383, 0.79296875],
           [0.7817383, 1., 0.7817383],
           [0.79296875, 0.7817383,  1.]])
    smiles = [
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([Br])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H]c1c(c(c(c(c1[H])Cl)[H])N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])N=C(N3[H])[H])[H]'
    ]
    df = pd.DataFrame({'smiles': smiles})
    sim_df = calculate_similarity(
        df, data_type='small_molecule', similarity_metric='fingerprint',
        field_name='smiles'
    )
    mtx = sim_df2mtx(sim_df).toarray()
    np.testing.assert_allclose(mtx, objective, rtol=0.001)
