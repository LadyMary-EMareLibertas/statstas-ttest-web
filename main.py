from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

# 프론트엔드에서 접근할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["https://your-vercel-app.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ttest")
async def run_ttest(request: Request):
    data = await request.json()
    before = np.array(data["before"])
    after = np.array(data["after"])

    if len(before) != len(after):
        return {"error": "Input arrays must have the same length"}

    diff = after - before
    n = len(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(n))

    return {
        "n": n,
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "t_stat": round(t_stat, 4)
    }