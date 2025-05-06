from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from scipy import stats

app = FastAPI()

# 프론트엔드에서 접근할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 구체적인 도메인으로 바꿔주는 것이 안전
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ttest/paired-two-tailed")
async def paired_ttest(request: Request):
    data = await request.json()
    before = np.array(data.get("before", []))
    after = np.array(data.get("after", []))
    alpha = data.get("alpha", 0.05)

    if len(before) != len(after):
        return {"error": "Input arrays must have the same length"}

    # 차이 계산
    diff = after - before

    # 정규성 검정
    shapiro_p = stats.shapiro(diff).pvalue
    ks_p = stats.kstest(diff, 'norm', args=(np.mean(diff), np.std(diff, ddof=1))).pvalue
    ad_result = stats.anderson(diff)
    ad_stat = ad_result.statistic
    ad_crit = ad_result.critical_values[2]  # 5% 유의수준 기준

    shapiro_pass = shapiro_p > alpha
    ks_pass = ks_p > alpha
    ad_pass = ad_stat < ad_crit
    normality_passed = shapiro_pass or ks_pass or ad_pass

    if not normality_passed:
        return {
            "error": "Normality assumption failed. Consider using Wilcoxon Signed-Rank test.",
            "shapiro_p": shapiro_p,
            "ks_p": ks_p,
            "ad_stat": ad_stat,
            "ad_crit": ad_crit
        }

    # t-검정
    t_stat, p_val = stats.ttest_rel(after, before)
    df = len(before) - 1
    critical_val = stats.t.ppf(1 - alpha / 2, df)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = abs(mean_diff / std_diff)

    return {
        "n": len(before),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_val, 4),
        "critical_value": round(critical_val, 4),
        "alpha": alpha,
        "significance": "significant" if p_val < alpha else "not significant",
        "cohens_d": round(cohens_d, 3),
        "normality": {
            "shapiro_p": round(shapiro_p, 4),
            "ks_p": round(ks_p, 4),
            "anderson_stat": round(ad_stat, 4),
            "anderson_crit": round(ad_crit, 4),
            "passed": normality_passed
        }
    }