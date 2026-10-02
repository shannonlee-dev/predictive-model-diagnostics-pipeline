"""중단된 CPU 탐색 작업을 재개하고 결과를 원자적으로 저장한다."""

from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from .training import trial


def run_jobs(jobs, output, workers):
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    keys = {
        (r["config_id"], r["seed"], r["train_end"], r["validation_end"]) for r in rows
    }
    total = len(jobs)
    jobs = [j for j in jobs if (j[0]["config_id"], j[1], j[2], j[3]) not in keys]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(trial, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            temporary = output.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(output)
            if len(rows) % 16 == 0 or len(rows) == total:
                print(f"{output.name}: {len(rows)}/{total} completed", flush=True)
    return pd.DataFrame(rows)
