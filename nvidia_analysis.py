import sqlite3
import sys
import os
import csv

def ns_to_ms(ns):
    return ns / 1e6

def table_exists(cur, table_name):
    cur.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table_name,))
    return cur.fetchone() is not None

def compute_overlap(intervals_a, intervals_b):
    """
    Compute total overlap between two interval lists.
    Each list contains (start, end) tuples in ns.
    """
    if not intervals_a or not intervals_b:
        return 0

    intervals_a = sorted(intervals_a)
    intervals_b = sorted(intervals_b)

    overlap = 0
    i = j = 0

    while i < len(intervals_a) and j < len(intervals_b):
        a_start, a_end = intervals_a[i]
        b_start, b_end = intervals_b[j]

        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start < end:
            overlap += end - start

        if a_end < b_end:
            i += 1
        else:
            j += 1

    return overlap


def main(sqlite_file):
    conn = sqlite3.connect(sqlite_file)
    cur = conn.cursor()

    cur.execute("""
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    """)
    gpu_compute_intervals = cur.fetchall()
    gpu_compute_ns = sum(e - s for s, e in gpu_compute_intervals)

    
    cur.execute("""
        SELECT start, end, copyKind
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
    """)
    memcpy_rows = cur.fetchall()

    memcpy_intervals = []
    memcpy_by_kind = {}

    memcpy_kind_map = {
        1: "HtoD",
        2: "DtoH",
        3: "DtoD",
        8: "PtoP",
        11:"Unified HtoD",
        12:"Unified DtoH"
    }

    for s, e, kind in memcpy_rows:
        memcpy_intervals.append((s, e))
        memcpy_by_kind.setdefault(kind, 0)
        memcpy_by_kind[kind] += (e - s)

    total_memcpy_ns = sum(memcpy_by_kind.values())

    cur.execute("SELECT start, end FROM NVTX_EVENTS")
    cpu_intervals = cur.fetchall()
    cpu_compute_ns = sum(e - s for s, e in cpu_intervals)

    if table_exists(cur, "OS_RUNTIME_API"):
        cur.execute("SELECT start, end FROM OS_RUNTIME_API")
        cpu_os = cur.fetchall()
        cpu_compute_ns += sum(e - s for s, e in cpu_os)
        cpu_intervals += cpu_os

  
    cpu_gpu_overlap_ns = compute_overlap(
        gpu_compute_intervals, cpu_intervals
    )

    gpu_memcpy_overlap_ns = compute_overlap(
        gpu_compute_intervals, memcpy_intervals
    )

    gpu_hidden_by_cpu_pct = (
        cpu_gpu_overlap_ns / gpu_compute_ns * 100
        if gpu_compute_ns else 0
    )

    cpu_overlapped_with_gpu_pct = (
        cpu_gpu_overlap_ns / cpu_compute_ns * 100
        if cpu_compute_ns else 0
    )

    memcpy_hidden_by_gpu_pct = (
        gpu_memcpy_overlap_ns / total_memcpy_ns * 100
        if total_memcpy_ns else 0
    )

  
    print("\n ===== Nsight Systems Overlap Summary =====\n")

    print("Absolute Times:")
    print(f"GPU Compute Time      : {ns_to_ms(gpu_compute_ns):.3f} ms")
    print(f"GPU Memcpy Time       : {ns_to_ms(total_memcpy_ns):.3f} ms")
    print(f"CPU Compute Time      : {ns_to_ms(cpu_compute_ns):.3f} ms\n")

    print("GPU Memcpy Breakdown:")
    for kind, t in memcpy_by_kind.items():
        name = memcpy_kind_map.get(kind, f"Unknown({kind})")
        print(f"  {name:8s}: {ns_to_ms(t):.3f} ms")

    print("Overlap Times:")
    print(f"CPU ↔ GPU Compute     : {ns_to_ms(cpu_gpu_overlap_ns):.3f} ms")
    print(f"GPU Compute ↔ Memcpy  : {ns_to_ms(gpu_memcpy_overlap_ns):.3f} ms\n")

    print("Overlap Percentages:")
    print(f"GPU hidden by CPU     : {gpu_hidden_by_cpu_pct:.2f} %")
    print(f"CPU overlapped w/ GPU : {cpu_overlapped_with_gpu_pct:.2f} %")
    print(f"Memcpy hidden by GPU  : {memcpy_hidden_by_gpu_pct:.2f} %")

    conn.close()

    base_name = os.path.basename(sqlite_file)
    name_no_ext = os.path.splitext(base_name)[0]
    csv_file = f"nsys_analysis_{name_no_ext}.csv"
    file_exists = os.path.isfile(csv_file)

    row = {
    "gpu_compute_ms": ns_to_ms(gpu_compute_ns),
    "gpu_memcpy_ms": ns_to_ms(total_memcpy_ns),
    "cpu_compute_ms": ns_to_ms(cpu_compute_ns),

    "memcpy_htod_ms": ns_to_ms(memcpy_by_kind.get(1, 0)),
    "memcpy_dtoh_ms": ns_to_ms(memcpy_by_kind.get(2, 0)),
    "memcpy_dtod_ms": ns_to_ms(memcpy_by_kind.get(3, 0)),
    "memcpy_p2p_ms":  ns_to_ms(memcpy_by_kind.get(8, 0)),
    "memcpy_unified_htod_ms":  ns_to_ms(memcpy_by_kind.get(11, 0)),
    "memcpy_unified_dtoh_ms":  ns_to_ms(memcpy_by_kind.get(12, 0)),

    "cpu_gpu_overlap_ms": ns_to_ms(cpu_gpu_overlap_ns),
    "gpu_memcpy_overlap_ms": ns_to_ms(gpu_memcpy_overlap_ns),

    "gpu_hidden_by_cpu_pct": gpu_hidden_by_cpu_pct,
    "cpu_overlapped_with_gpu_pct": cpu_overlapped_with_gpu_pct,
    "memcpy_hidden_by_gpu_pct": memcpy_hidden_by_gpu_pct,
     }

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":

    main(sys.argv[1])
