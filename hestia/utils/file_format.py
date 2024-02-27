import os
import math
import shutil
from typing import Dict, List


def _write_fasta(
    sequences: List[str],
    ids: List[str],
    filename: str
) -> None:
    with open(filename, 'w') as file:
        for sequence, id in zip(sequences, ids):
            file.write(f'>{id}\n{sequence}\n')


def _write_fasta_chunks(
    sequences: List[str],
    ids: List[str],
    n_chunks: int,
    tmp_dir: str
) -> int:
    chunk_size = math.ceil(len(ids) / n_chunks)
    if (n_chunks - 1) * chunk_size >= len(ids):
        n_chunks -= 1

    for i in range(n_chunks):
        chunk_ids = ids[i * chunk_size: (i+1) * chunk_size]
        chunk_seqs = sequences[i * chunk_size: (i+1) * chunk_size]
        path = os.path.join(tmp_dir, f'{i}.fasta.tmp')
        _write_fasta(chunk_seqs, chunk_ids, path)
    return n_chunks


def get_len_dict(ids: List[str], seqs: List[str]) -> Dict[str, int]:
    '''Get a dictionary that contains the length of each sequence.'''
    len_dict = {}
    for id, seq in zip(ids, seqs):
        len_dict[id] = len(seq)
    return len_dict


def _collect_pdb_files(pdb_paths: List[str], output_dir: str) -> None:
    for path in pdb_paths:
        new_path = os.path.join(output_dir, os.path.basename(path))
        shutil.copy(path, new_path)
