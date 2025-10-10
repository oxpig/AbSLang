from __future__ import annotations
from typing import List, Tuple

from anarci import validate_sequence as _anarci_validate_sequence, anarci as _anarci


def _number_single_sequence(
    sequence: str,
    chain: str,
    *,
    scheme: str = "imgt",
) -> List[Tuple[Tuple[int, str], str]]:
    """
    Number and validate a single chain using ANARCI and return IMGT-numbered residues.

    Returns a list of ((position, insertion_code), amino_acid) excluding gaps.

    Assumptions:
    - Heavy chains use chain="H"
    - Light chains use chain="L" (accepts both L and K)
    - No species restrictions
    """
    # Basic amino acid validation
    _anarci_validate_sequence(sequence)

    # Require plausible Ig variable domain length
    assert (
        len(sequence) > 70
    ), f"Sequence too short to be an Ig domain. Please provide whole sequence: {sequence}"

    # Configure allowed chains: L accepts both lambda (L) and kappa (K)
    allow = {chain}
    if chain == "L":
        allow.update({"K"})

    # IMGT scheme for numbering sanity checks
    numbered, _, _ = _anarci(
        [("sequence", sequence)],
        scheme="imgt",
        output=False,
        allow=allow,
        allowed_species=None,  # accept any species
    )

    assert numbered[0], (
        f"Sequence provided as a {chain} chain is not recognised as a {chain} chain."
    )

    # Extract non-gap positions
    output = [x for x in numbered[0][0][0] if x[1] != "-"]
    numbers = [x[0][0] for x in output]

    # Sanity check on coverage of variable domain (IMGT positions)
    assert (max(numbers) > 120) and (
        min(numbers) < 8
    ), "Sequence missing too many residues to model correctly. Please provide whole variable domain."

    if scheme == "raw":
        # Re-emit as contiguous raw numbering if requested
        output = [((i + 1, " "), x[1]) for i, x in enumerate(output)]
    elif scheme != "imgt":
        # Renumber with requested scheme
        numbered, _, _ = _anarci(
            [("sequence", sequence)],
            scheme=scheme,
            output=False,
            allow=allow,
            allowed_species=None,
        )
        output = [x for x in numbered[0][0][0] if x[1] != "-"]

    return output


def _trim_to_fv(sequence: str, chain: str) -> str:
    numbered = _number_single_sequence(sequence, chain, scheme="imgt")
    return "".join([aa for (_, aa) in numbered])


def validate_and_trim(sequence: str, mode: str) -> str:
    """
    Validate input sequence(s) and trim to Fv for use in inference.
    """
    mode = mode.lower()
    if mode == "paired":
        if "|" not in sequence:
            raise AssertionError(
                "Paired mode requires sequences formatted as 'Heavy|Light'"
            )
        heavy, light = sequence.split("|", maxsplit=1)
        heavy = heavy.strip()
        light = light.strip()
        h_fv = _trim_to_fv(heavy, "H")
        l_fv = _trim_to_fv(light, "L")
        return f"{h_fv}|{l_fv}"
    elif mode in ("hc", "nb"):
        seq = sequence.strip()
        if "|" in seq:
            raise AssertionError(
                f"Mode '{mode}' expects a single heavy chain sequence without '|'"
            )
        return _trim_to_fv(seq, "H")
    else:
        raise ValueError("mode must be one of 'paired', 'hc', or 'nb'")

