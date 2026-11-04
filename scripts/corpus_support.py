"""Expanded coefficient-free support invariant under spatial-axis permutations."""
import itertools
import sympy as sp
from math_tokenizer.eqgpt.corpus_conversion import COORDS,U,support_key

def invariant_support(expression):
    keys=[]
    for perm in itertools.permutations(COORDS[:3]):
        changed=expression.xreplace(dict(zip(COORDS[:3],perm)))
        # Coordinate permutation changes the formal function's arguments too.
        changed=changed.xreplace({sp.Function('u')(*perm,COORDS[3]):U})
        keys.append(support_key(changed))
    return min(keys)
