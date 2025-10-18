#!/bin/bash
set -euo pipefail

# 1) temp location
tmpdir=$(mktemp -d /tmp/opponentpp-XXXX.XXXXXXXXXX)
echo "Working in: $tmpdir"

# 2) enter temp, clone your Semantic-Lexicon repository
cd "$tmpdir"
git clone https://github.com/farukalpay/Semantic-Lexicon.git
cd Semantic-Lexicon

# 3) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 4) install your package with data extras
pip install --quiet ".[dev,docs]"

# 5) train the model
semantic-lexicon prepare \
  --intent src/semantic_lexicon/data/intent.jsonl \
  --knowledge src/semantic_lexicon/data/knowledge.jsonl \
  --workspace artifacts

semantic-lexicon train --workspace artifacts

# 6) run Python script that handles matrix operations
python - <<'PY'
import re
from pathlib import Path
from semantic_lexicon import NeuralSemanticModel, load_config
from semantic_lexicon.utils import normalise_text

PROMPT = """Given:
S = [[2, 3], [1, 4]]
R = [[1, 0], [2, 1]]
v = (1, 2)

Calculate RS·v and SR·v"""

def parse_matrix(label, text):
    # S=[[a,b],[c,d]] style
    m = re.search(
        rf"{label}\s*=\s*\[\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]\s*,\s*\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]\s*\]",
        text, re.IGNORECASE,
    )
    if not m: return None
    a,b,c,d = map(float, m.groups())
    return [[a,b],[c,d]]

def parse_vector(text):
    m = re.search(r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)", text)
    return None if not m else (float(m.group(1)), float(m.group(2)))

def mm(A,B):
    return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
            [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]

def mv(M,v):
    return (M[0][0]*v[0]+M[0][1]*v[1], M[1][0]*v[0]+M[1][1]*v[1])

def fmt_num(x): return str(int(x)) if float(x).is_integer() else f"{x:.4g}"
def fmt_mat(M):
    # ASCII matrix block
    return f"[ {fmt_num(M[0][0]).rjust(4)} {fmt_num(M[0][1]).rjust(4)} ]\n[ {fmt_num(M[1][0]).rjust(4)} {fmt_num(M[1][1]).rjust(4)} ]"
def fmt_vec(v): return f"( {fmt_num(v[0])}, {fmt_num(v[1])} )"

S = parse_matrix("S", PROMPT)
R = parse_matrix("R", PROMPT)
v = parse_vector(PROMPT)
can_compute = S is not None and R is not None and v is not None

if can_compute:
    RS, SR = mm(R,S), mm(S,R)
    RSv, SRv = mv(RS,v), mv(SR,v)
    print("## Matrices")
    print("S =\n" + fmt_mat(S))
    print("\nR =\n" + fmt_mat(R))
    print("\n## Composition")
    print("RS = R × S =\n" + fmt_mat(RS))
    print("\nSR = S × R =\n" + fmt_mat(SR))
    print("\n## Results")
    print("v =", fmt_vec(v))
    print("RS · v =", fmt_vec(RSv))
    print("SR · v =", fmt_vec(SRv))
else:
    # fallback: use the library to answer normally
    model, _ = (NeuralSemanticModel(), load_config(None))
    artifacts = Path("artifacts")
    if (artifacts / "embeddings.json").exists():
        model = NeuralSemanticModel.load(artifacts, config=model.config)
    result = model.generate(normalise_text(PROMPT), persona=None)
    print(result.response)
PY

# 7) cleanup
deactivate
cd ~
rm -rf "$tmpdir"
echo "Cleaned up: $tmpdir"
