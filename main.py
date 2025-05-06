from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# ✅ JSON 요청 바디 구조 정의
class TTestInput(BaseModel):
    before: list[float]
    after: list[float]

@app.post("/ttest")
def ttest(data: TTestInput):
    before = np.array(data.before)
    after = np.array(data.after)

    if len(before) != len(after):
        return {"error": "Length mismatch"}

    diff = after - before
    n = len(diff)
    mean = np.mean(diff)
    std = np.std(diff, ddof=1)
    t = mean / (std / np.sqrt(n))

    return {
        "n": n,
        "mean_diff": round(mean, 2),
        "std_diff": round(std, 2),
        "t_stat": round(t, 4)
    }