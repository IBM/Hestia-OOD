from hestia.similarity import (molecular_similarity,
                               embedding_similarity, sim_df2mtx)
import numpy as np
import pandas as pd


def test_fingerprint_alignment():
    smiles = [
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([Br])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H]c1c(c(c(c(c1[H])Cl)[H])N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])N=C(N3[H])[H])[H]'
    ]
    df = pd.DataFrame({'smiles': smiles})
    sim_df = molecular_similarity(
        df, fingerprint='ecfp',
        sim_function='tanimoto',
        field_name='smiles'
    )
    objective_df = pd.DataFrame({
        'query': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'target': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'metric': [1., 0.773585, 0.785714, 0.773585, 1.,
                   0.773585, 0.785714, 0.773585, 1.]
    })
    assert sim_df['query'].to_list() == objective_df['query'].tolist()
    assert sim_df['target'].to_list() == objective_df['target'].tolist()
    np.testing.assert_allclose(sim_df['metric'].to_list(),
                               objective_df['metric'].tolist(), rtol=0.001)

    sim_df2 = molecular_similarity(
        df, fingerprint='ecfp',
        field_name='smiles', threshold=0.8,
        sim_function='tanimoto'
    )

    objective_df2 = pd.DataFrame({
        'query': [0, 1, 2],
        'target': [0, 1, 2],
        'metric': [1.0, 1.0, 1.0]
    })
    assert sim_df2['query'].to_list() == objective_df2['query'].tolist()
    assert sim_df2['target'].to_list() == objective_df2['target'].tolist()
    np.testing.assert_allclose(sim_df2['metric'].to_numpy(),
                               objective_df2['metric'].to_numpy(), rtol=0.001)


def test_embedding_cosine_similarity():
    a = np.array([0.8878814280709818, 0.44811583848916037,
                  0.8581227306329297, 0.7260848138819995,
                  0.828707815386661, 0.42991477001588163,
                  0.10240789838230091, 0.23037585728800736,
                  0.48593961297193566, 0.4555966011657775,
                  0.45357605892883157, 0.34027309465363187])
    b = np.array([0.588989182929779, 0.07390301922086917,
                  0.4333103070847615, 0.36951450755069026,
                  0.08558155244426213, 0.7975081300964375,
                  0.18330253332252744, 0.2948362342792936,
                  0.9012307123359792, 0.03047035688324251,
                  0.34864062629605075, 0.4795065290549264])
    df_query = np.stack([a, b])
    sim_df = embedding_similarity(df_query, sim_function='cosine-np')
    objective_df = pd.DataFrame({
        'query': [0, 0, 1, 1],
        'target': [0, 1, 0, 1],
        'metric': [1., 0.76962, 0.76962, 1.]
    })
    assert sim_df['query'].to_list() == objective_df['query'].tolist()
    assert sim_df['target'].to_list() == objective_df['target'].tolist()
    np.testing.assert_allclose(sim_df['metric'].to_numpy(), objective_df.metric,
                               rtol=0.001)


def test_embedding_euclidean_similarity():
    a = np.array([0.8878814280709818, 0.44811583848916037,
                  0.8581227306329297, 0.7260848138819995,
                  0.828707815386661, 0.42991477001588163,
                  0.10240789838230091, 0.23037585728800736,
                  0.48593961297193566, 0.4555966011657775,
                  0.45357605892883157, 0.34027309465363187])
    b = np.array([0.588989182929779, 0.07390301922086917,
                  0.4333103070847615, 0.36951450755069026,
                  0.08558155244426213, 0.7975081300964375,
                  0.18330253332252744, 0.2948362342792936,
                  0.9012307123359792, 0.03047035688324251,
                  0.34864062629605075, 0.4795065290549264])
    df_query = np.stack([a, b])
    sim_df = embedding_similarity(df_query, sim_function='euclidean')
    print(sim_df.head())
    objective_df = pd.DataFrame({
        'query': [0, 0, 1, 1],
        'target': [0, 1, 0, 1],
        'metric': [1.,  0.440091, 0.440091, 1.]
    })
    assert sim_df['query'].to_list() == objective_df['query'].tolist()
    assert sim_df['target'].to_list() == objective_df.target.tolist()
    np.testing.assert_allclose(sim_df['metric'].to_numpy(), objective_df.metric,
                               rtol=0.001)


def test_mapchiral():
    smiles = [
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([Br])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H]c1c(c(c(c(c1[H])Cl)[H])N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])N=C(N3[H])[H])[H]'
    ]
    df = pd.DataFrame({'smiles': smiles})
    sim_df = molecular_similarity(
        df, fingerprint='mapc',
        sim_function='jaccard',
        field_name='smiles',
        threshold=0.
    )
    objective_df = pd.DataFrame({
        'query': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'target': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'metric': [1.0, 0.5634765625, 0.0390625,
                   0.5634765625, 1.0, 0.0302734375,
                   0.0390625, 0.0302734375, 1.0]
    })
    assert sim_df['query'].to_list() == objective_df['query'].tolist()
    assert sim_df['target'].to_list() == objective_df['target'].tolist()
    np.testing.assert_allclose(sim_df['metric'].to_list(),
                               objective_df['metric'].tolist(), rtol=0.001)


def test_maccs():
    smiles = [
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([Br])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H]c1c(c(c(c(c1[H])Cl)[H])N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])N=C(N3[H])[H])[H]'
    ]
    df = pd.DataFrame({'smiles': smiles})
    sim_df = molecular_similarity(
        df, fingerprint='maccs',
        sim_function='tanimoto',
        field_name='smiles'
    )
    objective_df = pd.DataFrame({
        'query': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'target': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'metric': [1.,  0.921569, 0.961538, 0.921569, 1.,
                   0.921569, 0.961538, 0.921569, 1.]
    })

    assert sim_df['query'].to_list() == objective_df['query'].tolist()
    assert sim_df['target'].to_list() == objective_df['target'].tolist()
    np.testing.assert_allclose(sim_df['metric'].to_list(),
                               objective_df['metric'].tolist(), rtol=0.001)


def test_simdf2mtx():
    objective = np.array(
         [[1., 0.773585, 0.785714],
          [0.773585, 1., 0.773585],
          [0.785714, 0.773585, 1.]])
    smiles = [
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([Br])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H][C]1=[N][C]2=[C]([O][C]([H])([H])[C]3([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]([H])([H])[C]3([H])[H])[N]=[C]([N]([H])[C]3=[C]([H])[C]([H])=[C]([H])[C]([H])=[C]3[H])[N]=[C]2[N]1[H]',
        '[H]c1c(c(c(c(c1[H])Cl)[H])N([H])c2nc3c(c(n2)OC([H])([H])C4(C(C(C(C(C4([H])[H])([H])[H])([H])[H])([H])[H])([H])[H])[H])N=C(N3[H])[H])[H]'
    ]
    df = pd.DataFrame({'smiles': smiles})
    sim_df = molecular_similarity(
        df, fingerprint='ecfp',
        field_name='smiles', sim_function='tanimoto'
    )
    mtx = sim_df2mtx(sim_df, boolean_out=False).toarray()
    np.testing.assert_allclose(mtx, objective, rtol=0.001)
