"""Convert the supplied ASCII STL to binary (MuJoCo only reads binary STL)."""
import os
import shutil
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "robot", "assets", "lamp_shade.stl")
BACKUP = os.path.join(ROOT, "robot", "assets", "lamp_shade_ascii.stl")

if not os.path.exists(BACKUP):
    shutil.copy(SRC, BACKUP)           # keep the original untouched

mesh = trimesh.load(BACKUP, force="mesh")
print(f"faces: {len(mesh.faces)}  vertices: {len(mesh.vertices)}")
print(f"size (x, y, z): {mesh.extents.round(4)}")   # should be ~0.1–0.25 if units are metres

mesh.export(SRC, file_type="stl")      # trimesh writes binary STL by default
print(f"wrote binary STL → {SRC}")