"""
优化算法对比模块 — PSO / GA / EGPSO 多算法横向对比 + 两阶段鲁棒优化框架

理论来源: 文件08_系统容量优化配置.md
  - §1.1 主要优化方法对比 (两阶段鲁棒/双层/多目标/NSGA-III/EGPSO)
  - §2.1 算法特性对比
  - §2.2 EGPSO算法要点

与现有 capacity_optimization.py (PSO) 和 nsga2.py (NSGA-II) 互补:
  - 新增GA, EGPSO实现
  - 新增两阶段鲁棒优化框架
  - 多算法统一接口对比

运行:
    python optimization_comparison.py   # 模块自测
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SERVICE_AREA_CONFIG, PV_AREA_RATIO,
)


# ============================================================
# 1. 通用优化器接口
# ============================================================

class OptimizerBase:
    """优化器基类 — 统一接口"""

    def __init__(self, evaluator, bounds, seed=42):
        """
        evaluator: callable(x) -> float (fitness, 越小越好)
        bounds: np.ndarray (d, 2), 决策变量上下界
        """
        self.evaluator = evaluator
        self.bounds = np.array(bounds)
        self.dim = self.bounds.shape[0]
        self.rng = np.random.RandomState(seed)
        self.history = []  # [(gen, best_fitness, best_x)]

    def _clip(self, x):
        """边界约束"""
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def evaluate(self, x):
        """评价 (含边界惩罚)"""
        x_clipped = self._clip(x)
        try:
            return self.evaluator(x_clipped)
        except Exception:
            return 1e12


# ============================================================
# 2. 遗传算法 (GA)
# ============================================================

class GeneticAlgorithm(OptimizerBase):
    """标准遗传算法

    操作:
    - 轮盘赌选择 (Roulette Wheel Selection)
    - 模拟二进制交叉 (SBX)
    - 多项式变异
    - 精英保留
    """

    def __init__(self, evaluator, bounds, pop_size=60, n_gen=50,
                 crossover_prob=0.9, mutation_prob=0.1,
                 eta_c=20, eta_m=20, elite_size=2, seed=42):
        super().__init__(evaluator, bounds, seed=seed)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.elite_size = elite_size

    def _sbx_crossover(self, p1, p2):
        """模拟二进制交叉"""
        if self.rng.random() > self.pc:
            return p1.copy(), p2.copy()

        c1, c2 = np.zeros(self.dim), np.zeros(self.dim)
        for i in range(self.dim):
            if self.rng.random() < 0.5:
                if abs(p2[i] - p1[i]) > 1e-10:
                    if p1[i] < p2[i]:
                        y1, y2 = p1[i], p2[i]
                    else:
                        y1, y2 = p2[i], p1[i]
                    yl, yu = self.bounds[i, 0], self.bounds[i, 1]

                    beta = 1.0 + 2.0 * (y1 - yl) / (y2 - y1 + 1e-10)
                    alpha = 2.0 - beta ** (-(self.eta_c + 1))
                    u = self.rng.random()
                    if u <= 1.0 / alpha:
                        betaq = (u * alpha) ** (1.0 / (self.eta_c + 1))
                    else:
                        betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1))

                    c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                else:
                    c1[i], c2[i] = p1[i], p2[i]
            else:
                c1[i], c2[i] = p1[i], p2[i]

        return self._clip(c1), self._clip(c2)

    def _polynomial_mutation(self, x):
        """多项式变异"""
        for i in range(self.dim):
            if self.rng.random() < self.pm:
                yl, yu = self.bounds[i, 0], self.bounds[i, 1]
                delta = min(x[i] - yl, yu - x[i]) / (yu - yl + 1e-10)
                u = self.rng.random()
                if u < 0.5:
                    delta_q = (2 * u + (1 - 2 * u) * (1 - delta) ** (self.eta_m + 1)) ** \
                              (1.0 / (self.eta_m + 1)) - 1.0
                else:
                    delta_q = 1.0 - (2 * (1 - u) + 2 * (u - 0.5) *
                                     (1 - delta) ** (self.eta_m + 1)) ** \
                              (1.0 / (self.eta_m + 1))
                x[i] += delta_q * (yu - yl)
        return self._clip(x)

    def optimize(self, verbose=False):
        """执行GA优化"""
        # 初始化
        pop = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            pop[i] = [self.rng.uniform(*self.bounds[j]) for j in range(self.dim)]
        fitness = np.array([self.evaluate(p) for p in pop])

        best_idx = np.argmin(fitness)
        best_x, best_f = pop[best_idx].copy(), fitness[best_idx]

        for gen in range(self.n_gen):
            # 精英选择
            elite_idx = np.argsort(fitness)[:self.elite_size]
            new_pop = [pop[i].copy() for i in elite_idx]

            # 轮盘赌选择 + 交叉 + 变异
            epsilon = 1e-10
            fit_adj = 1.0 / (fitness - fitness.min() + epsilon)
            fit_probs = fit_adj / fit_adj.sum()

            while len(new_pop) < self.pop_size:
                # 选择
                parents = self.rng.choice(self.pop_size, size=2, p=fit_probs, replace=False)
                p1, p2 = pop[parents[0]], pop[parents[1]]
                # 交叉
                c1, c2 = self._sbx_crossover(p1, p2)
                # 变异
                c1 = self._polynomial_mutation(c1)
                c2 = self._polynomial_mutation(c2)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = np.array(new_pop[:self.pop_size])
            fitness = np.array([self.evaluate(p) for p in pop])

            gen_best = np.argmin(fitness)
            if fitness[gen_best] < best_f:
                best_x, best_f = pop[gen_best].copy(), fitness[gen_best]

            self.history.append((gen, best_f, best_x.copy()))
            if verbose and gen % 10 == 0:
                print(f"  GA Gen {gen:3d}: best={best_f:.2e}")

        return best_x, best_f, self.history


# ============================================================
# 3. EGPSO (Elite Genetic PSO) — 文件08 §2.2
# ============================================================

class EGPSO(OptimizerBase):
    """精英遗传粒子群优化 (Elite Genetic PSO)

    改进:
    - 自适应惯性权重 (线性递减)
    - 精英保留 (每代最优k个直接进入下一代)
    - 精英交叉变异 (精英粒子间SBX交叉)
    - 速度限幅
    """

    def __init__(self, evaluator, bounds, pop_size=50, n_gen=40,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0,
                 elite_size=3, crossover_prob=0.5, mutation_prob=0.1, seed=42):
        super().__init__(evaluator, bounds, seed=seed)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.elite_size = elite_size
        self.pc = crossover_prob
        self.pm = mutation_prob

    def _inertia_weight(self, gen):
        """自适应惯性权重 (线性递减)"""
        return self.w_start - (self.w_start - self.w_end) * gen / max(self.n_gen - 1, 1)

    def optimize(self, verbose=False):
        """执行EGPSO优化"""
        d = self.dim
        # 速度边界
        v_max = 0.2 * (self.bounds[:, 1] - self.bounds[:, 0])
        v_min = -v_max

        # 初始化
        pop = np.zeros((self.pop_size, d))
        vel = np.zeros((self.pop_size, d))
        for i in range(self.pop_size):
            pop[i] = [self.rng.uniform(*self.bounds[j]) for j in range(d)]
            vel[i] = [self.rng.uniform(v_min[j], v_max[j]) for j in range(d)]

        fitness = np.array([self.evaluate(p) for p in pop])
        p_best = pop.copy()
        p_best_f = fitness.copy()
        g_best_idx = np.argmin(fitness)
        g_best, g_best_f = pop[g_best_idx].copy(), fitness[g_best_idx]

        for gen in range(self.n_gen):
            w = self._inertia_weight(gen)

            # 精英保留
            elite_idx = np.argsort(fitness)[:self.elite_size]
            elites = pop[elite_idx].copy()

            # PSO更新
            for i in range(self.pop_size):
                r1, r2 = self.rng.rand(d), self.rng.rand(d)
                vel[i] = (w * vel[i] +
                          self.c1 * r1 * (p_best[i] - pop[i]) +
                          self.c2 * r2 * (g_best - pop[i]))
                vel[i] = np.clip(vel[i], v_min, v_max)
                pop[i] = self._clip(pop[i] + vel[i])

            # 精英交叉 (SBX)
            for k in range(min(self.elite_size, len(elites) - 1)):
                if self.rng.random() < self.pc:
                    idx1 = elite_idx[k]
                    idx2 = elite_idx[(k + 1) % self.elite_size]
                    c1, c2 = self._sbx_crossover(pop[idx1], pop[idx2])
                    pop[idx1], pop[idx2] = c1, c2

            # 精英变异
            for k in elite_idx[:self.elite_size]:
                if self.rng.random() < self.pm:
                    for j in range(d):
                        if self.rng.random() < 0.3:
                            delta = self.rng.normal(0, 0.1) * (self.bounds[j, 1] - self.bounds[j, 0])
                            pop[k, j] = self._clip(pop[k, j] + delta)[j]

            # 评价
            fitness = np.array([self.evaluate(p) for p in pop])

            # 更新p_best
            improved = fitness < p_best_f
            p_best[improved] = pop[improved]
            p_best_f[improved] = fitness[improved]

            # 更新g_best
            gen_best = np.argmin(fitness)
            if fitness[gen_best] < g_best_f:
                g_best, g_best_f = pop[gen_best].copy(), fitness[gen_best]

            self.history.append((gen, g_best_f, g_best.copy()))
            if verbose and gen % 10 == 0:
                print(f"  EGPSO Gen {gen:3d}: best={g_best_f:.2e}")

        return g_best, g_best_f, self.history

    def _sbx_crossover(self, p1, p2):
        """简化的SBX交叉"""
        if self.rng.random() > self.pc:
            return p1.copy(), p2.copy()
        # 简化为均匀交叉+扰动
        alpha = self.rng.uniform(-0.25, 1.25, size=self.dim)
        c1 = self._clip(p1 + alpha * (p2 - p1))
        c2 = self._clip(p2 - alpha * (p2 - p1))
        return c1, c2


# ============================================================
# 4. 两阶段鲁棒优化框架 (文件08 §1.2)
# ============================================================

class RobustOptimizationFramework:
    """两阶段鲁棒优化框架

    文件08核心方法:
    外层(min): 投资决策 x = [PV_cap, ESS_cap, ESS_power]
    内层(max-min): 在最恶劣源荷场景下找出最优运行调度

    应用NC&CG (Nested Column and Constraint Generation) 思想,
    简化为: 枚举最恶劣场景 + 内层TOU调度优化

    Reference:
      中国电机工程学报, 2025: 高速公路服务区微网容量配置两阶段鲁棒优化
    """

    def __init__(self, evaluator, nominal_pv_coeff, nominal_load):
        """
        evaluator: callable(pv_cap, ess_cap, ess_pow, pv_coeff, load) -> dict
        nominal_pv_coeff: 标称光伏出力系数 (24h)
        nominal_load: 标称负荷 (24h)
        """
        self.evaluator = evaluator
        self.nominal_pv = np.array(nominal_pv_coeff)
        self.nominal_load = np.array(nominal_load)

    def worst_case_scenario(self, uncertainty_pv=0.20, uncertainty_load=0.15):
        """生成最恶劣场景: PV低出力 + 负荷高出力 + 峰谷电价差最大"""
        worst_pv = self.nominal_pv * (1 - uncertainty_pv)  # PV降20%
        worst_load = self.nominal_load * (1 + uncertainty_load)  # 负荷增15%
        return worst_pv, worst_load

    def robust_evaluate(self, pv_cap, ess_cap, ess_pow,
                        uncertainty_pv=0.20, uncertainty_load=0.15):
        """鲁棒评估: 在最恶劣场景下运行, 得到最保守的性能指标"""
        worst_pv, worst_load = self.worst_case_scenario(uncertainty_pv,
                                                         uncertainty_load)
        # 最恶劣场景
        worst_result = self.evaluator(pv_cap, ess_cap, ess_pow,
                                       worst_pv, worst_load)
        # 标称场景
        nominal_result = self.evaluator(pv_cap, ess_cap, ess_pow,
                                         self.nominal_pv, self.nominal_load)

        # 鲁棒性指标: 最恶劣场景性能/标称性能
        ssr_ratio = (worst_result.get('self_sufficiency', 0) /
                     max(nominal_result.get('self_sufficiency', 0.01), 0.01))
        npc_penalty = (worst_result.get('npc', 0) -
                       nominal_result.get('npc', 0))

        return {
            'nominal': nominal_result,
            'worst_case': worst_result,
            'robustness_ssr_ratio': ssr_ratio,
            'robustness_npc_penalty': npc_penalty,
            'worst_pv_coeff': worst_pv.tolist(),
            'worst_load': worst_load.tolist(),
        }

    def robust_optimize(self, candidate_configs, uncertainty_pv=0.20,
                        uncertainty_load=0.15):
        """在候选配置集中进行鲁棒择优

        candidate_configs: list of (pv_cap, ess_cap, ess_pow)
        返回按鲁棒NPC排序的最优配置
        """
        results = []
        for pv, ess_e, ess_p in candidate_configs:
            r = self.robust_evaluate(pv, ess_e, ess_p,
                                     uncertainty_pv, uncertainty_load)
            # 鲁棒目标: min max NPC (worst-case NPC)
            robust_npc = r['worst_case'].get('npc', 1e12)
            results.append({
                'pv': pv, 'ess_e': ess_e, 'ess_p': ess_p,
                'robust_npc': robust_npc,
                'nominal_npc': r['nominal'].get('npc', 0),
                'robustness_ssr': r['robustness_ssr_ratio'],
                'full': r,
            })

        results.sort(key=lambda x: x['robust_npc'])
        return results


# ============================================================
# 5. 算法对比工具
# ============================================================

class AlgorithmBenchmark:
    """多算法基准测试"""

    def __init__(self, evaluator, bounds, seed=42):
        self.evaluator = evaluator
        self.bounds = bounds
        self.seed = seed

    def run_comparison(self, methods=None, n_runs=3):
        """运行多算法对比"""
        if methods is None:
            methods = ['PSO', 'GA', 'EGPSO']

        results = {}
        for method in methods:
            print(f"\n  Running {method}...")
            runs = []
            for run in range(n_runs):
                t0 = time.time()
                if method == 'PSO':
                    from capacity_optimization import MicrogridOptimizer
                    # PSO是预先集成的, 这里跳过
                    pass
                elif method == 'GA':
                    opt = GeneticAlgorithm(self.evaluator, self.bounds,
                                           pop_size=50, n_gen=30,
                                           seed=self.seed + run)
                    best_x, best_f, history = opt.optimize()
                    elapsed = time.time() - t0
                elif method == 'EGPSO':
                    opt = EGPSO(self.evaluator, self.bounds,
                                pop_size=50, n_gen=30,
                                seed=self.seed + run)
                    best_x, best_f, history = opt.optimize()
                    elapsed = time.time() - t0

                runs.append({
                    'best_x': best_x,
                    'best_f': best_f,
                    'elapsed': elapsed,
                    'history': history,
                })

            if runs:
                best_run = min(runs, key=lambda r: r['best_f'])
                results[method] = {
                    'best_f': best_run['best_f'],
                    'best_x': best_run['best_x'],
                    'mean_f': np.mean([r['best_f'] for r in runs]),
                    'std_f': np.std([r['best_f'] for r in runs]),
                    'mean_time': np.mean([r['elapsed'] for r in runs]),
                    'convergence': best_run['history'],
                }

        return results

    def print_comparison(self, results):
        """打印算法对比表"""
        print("\n" + "=" * 65)
        print("优化算法横向对比")
        print("=" * 65)
        print(f"{'算法':<10} {'最优值':>10} {'均值':>10} {'标准差':>10} {'耗时(s)':>10}")
        print("-" * 52)
        for method, res in results.items():
            print(f"  {method:<8} {res['best_f']:>10.2e} {res['mean_f']:>10.2e} "
                  f"{res['std_f']:>10.2e} {res['mean_time']:>8.1f}")

        # 收敛曲线对比
        print(f"\n--- 收敛速度对比 (末代最优值) ---")
        for method, res in results.items():
            n_evals = len(res['convergence'])
            print(f"  {method}: {n_evals}代 → 终值={res['convergence'][-1][1]:.2e}")

    def get_algorithm_recommendations(self, results):
        """算法推荐"""
        recs = {}
        for method, res in results.items():
            conv = res['convergence']
            gen_80pct = next((i for i, (g, f, _) in enumerate(conv)
                              if f <= res['best_f'] * 1.1), len(conv))
            recs[method] = {
                'speed': 'fast' if gen_80pct < 15 else 'medium' if gen_80pct < 30 else 'slow',
                'stability': 'high' if res['std_f'] < abs(res['mean_f']) * 0.05 else 'medium',
                'global_search': 'strong' if res['std_f'] < abs(res['mean_f']) * 0.1 else 'moderate',
            }
        return recs


def self_test():
    """模块自测 (使用简化的测试函数)"""

    def test_evaluator(x):
        """Rosenbrock函数 (d=3)"""
        pv, ess_e, ess_p = x
        a = (1 - pv) ** 2 + 100 * (ess_e - pv ** 2) ** 2
        b = (ess_p - 0.5) ** 2 * 1000
        return a + b

    bounds = np.array([
        [0.0, 3.0],
        [0.0, 3.0],
        [0.0, 1.0],
    ])

    print("=" * 55)
    print("优化算法对比 — 自测 (Rosenbrock函数)")
    print("=" * 55)

    bench = AlgorithmBenchmark(test_evaluator, bounds, seed=42)
    results = bench.run_comparison(['GA', 'EGPSO'], n_runs=2)
    bench.print_comparison(results)

    recommendations = bench.get_algorithm_recommendations(results)
    print(f"\n--- 算法推荐 ---")
    for method, rec in recommendations.items():
        print(f"  {method}: speed={rec['speed']}, stability={rec['stability']}, "
              f"global={rec['global_search']}")


if __name__ == '__main__':
    self_test()
