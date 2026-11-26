"""Registers, spills, scratch and shared memory of the kernels in an exe or .so built by HIP (AMD, uncompressed bundles).
  python kregs.py <llvm-readelf of the ROCm SDK> <exe> [part of a kernel name ...]"""
import os, re, struct, subprocess, sys, tempfile
b = open(sys.argv[2], "rb").read()
td = tempfile.mkdtemp()
pos, n = 0, 0
if b"CCOB" in b:
    print("(compressed bundles in this exe: not read)")
while True:
    p = b.find(b"\x7fELF\x02\x01\x01\x40", pos)   # (an AMDGPU HSA code object)
    if p < 0:
        break
    shoff, = struct.unpack_from("<Q", b, p + 40)
    shent, shnum = struct.unpack_from("<HH", b, p + 58)
    size = shoff + shent * shnum
    f = os.path.join(td, f"{n}.o")
    open(f, "wb").write(b[p:p + size])
    o = subprocess.run([sys.argv[1], "--notes", f], capture_output=True, text=True).stdout
    tgt = re.search(r"amdhsa.target:\s*(\S+)", o)
    for k in re.split(r"\n  - ", o):
        m = re.search(r"\.name:\s+(\S+)\n", k[k.find(".max_flat"):] if ".max_flat" in k else k)
        if not m or not all(x in m[1] for x in sys.argv[3:]):
            continue
        g = {q: (re.search(r"\.%s:\s+(\d+)" % q, k) or [0, "?"])[1] for q in
             ("vgpr_count", "sgpr_count", "vgpr_spill_count", "private_segment_fixed_size", "group_segment_fixed_size")}
        print((tgt[1] if tgt else "?").split("-")[-1], m[1][:90], g)
    pos, n = p + 64, n + 1
print("code objects", n)
