"""
NSGA-II 多目标优化模块 — 非支配排序遗传算法

理论来源: 文件08_系统容量优化配置.md
  - §1.1 主要优化方法对比 (NSGA-III用于低碳场景多目标)
  - §1.2 典型优化模型数学表述 (两阶段鲁棒/双层优化)
  - §2.1 算法特性对比 (NSGA-III推荐用于>=3目标优化)

算法: NSGA-II (Deb et al., 2002)
  - 非支配排序 (Non-dominated Sorting)
  - 拥挤距离 (Crowding Distance)
  - 锦标赛选择 (Tournament Selection)
  - 模拟二进制交叉 (SBX, Simulated Binary Crossover)
  - 多项式变异 (Polynomial Mutation)

决策变量:
  x[0] = PV容量 (kWp)  — 连续
  x[1] = ESS容量 (kWh) — 连续
  x[2] = ESS功率 (kW)  — 连续

目标函数:
  f[0] = NPC (全生命周期净现值成本, 元) — min
  f[1] = 1 - 自洽率 (SSR)               — min (等价于max SSR)
  f[2] = -年碳减排量 (tCO2)               — min (等价于max carbon)
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    SERVICE_AREA_CONFIG, PV_AREA_RATIO,
)


class NSGA2:
    """NSGA-II 多目标优化器

    Parameters
    ----------
    evaluator : callable
        评价函数, 签名为 evaluator(pv_cap, ess_cap, ess_pow) -> dict
        返回字典必须包含: 'npc', 'self_sufficiency', 'carbon_reduction_t'
    bounds : np.ndarray (3, 2)
        决策变量上下界
    pop_size : int
        种群大小 (需为偶数)
    n_gen : int
        进化代数
    crossover_prob : float
        交叉概率
    mutation_prob : float
        变异概率
    eta_c : float
        SBX交叉分布指数 (越大子代越接近父代)
    eta_m : float
        多项式变异分布指数
    seed : int or None
        随机种子
    """

    def __init__(self, evaluator, bounds, pop_size=100, n_gen=50,
                 crossover_prob=0.9, mutation_prob=0.3,
                 eta_c=20, eta_m=20, seed=None):
        self.evaluator = evaluator
        self.bounds = np.asarray(bounds)
        self.pop_size = pop_size + (pop_size % 2)  # 确保偶数
        self.n_gen = n_gen
        self.p_cross = crossover_prob
        self.p_mut = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.rng = np.random.RandomState(seed)

        self.n_var = len(bounds)
        self.n_obj = 3  # NPC, 1-SSR, -Carbon

    def _initialize(self):
        """初始化种群: 拉丁超立方 + 随机混合"""
        pop = np.zeros((self.pop_size, self.n_var))
        for i in range(self.pop_size):
            for j in range(self.n_var):
                pop[i, j] = self.rng.uniform(self.bounds[j, 0], self.bounds[j, 1])
            # ESS功率约束: P_ess <= E_ess * 0.5
            pop[i, 2] = min(pop[i, 2], pop[i, 1] * 0.5)
        return pop

    def _evaluate_population(self, pop):
        """评估整个种群"""
        objectives = np.zeros((self.pop_size, self.n_obj))
        results_cache = []
        for i in range(self.pop_size):
            pv, ess_e, ess_p = pop[i]
            if ess_e < 10:
                ess_e = 0.0
                ess_p = 0.0
            ess_p = min(ess_p, ess_e * 0.5)

            try:
                result = self.evaluator(pv, ess_e, ess_p)
                results_cache.append(result)
                # f0: NPC (min)
                objectives[i, 0] = result['npc']
                # f1: 1 - SSR (min)
                objectives[i, 1] = 1.0 - result['self_sufficiency']
                # f2: -carbon (min → max carbon)
                objectives[i, 2] = -result['carbon_reduction_t']
            except Exception:
                # 评价失败 → 惩罚
                objectives[i, 0] = 1e12
                objectives[i, 1] = 1.0
                objectives[i, 2] = 0.0
                results_cache.append(None)

        return objectives, results_cache

    @staticmethod
    def _non_dominated_sort(objectives):
        """非支配排序: 返回 fronts (list of lists of indices)

        对于最小化问题:
         解A支配解B 当且仅当:
           f_i(A) <= f_i(B) for all i, AND
           f_j(A) < f_j(B) for at least one j
        """
        pop_size = len(objectives)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions = [[] for _ in range(pop_size)]

        for i in range(pop_size):
            for j in range(pop_size):
                if i == j:
                    continue
                # 检查i是否支配j
                i_dominates_j = True
                for k in range(objectives.shape[1]):
                    if objectives[i, k] > objectives[j, k]:
                        i_dominates_j = False
                        break
                if i_dominates_j:
                    # check strict inequality
                    strictly_better = False
                    for k in range(objectives.shape[1]):
                        if objectives[i, k] < objectives[j, k]:
                            strictly_better = True
                            break
                    if strictly_better:
                        dominated_solutions[i].append(j)
                    elif not strictly_better:
                        pass  # identical
                else:
                    # check if j dominates i
                    j_dominates_i = True
                    for k in range(objectives.shape[1]):
                        if objectives[j, k] > objectives[i, k]:
                            j_dominates_i = False
                            break
                    if j_dominates_i:
                        strictly_better = False
                        for k in range(objectives.shape[1]):
                            if objectives[j, k] < objectives[i, k]:
                                strictly_better = True
                                break
                        if strictly_better:
                            domination_count[i] += 1

        # 构建前沿
        fronts = []
        front_indices = set()
        while len(front_indices) < pop_size:
            current_front = []
            for i in range(pop_size):
                if i not in front_indices and domination_count[i] == 0:
                    current_front.append(i)
            if not current_front:
                # 剩余未排序的解
                remaining = [i for i in range(pop_size) if i not in front_indices]
                current_front = remaining
                fronts.append(current_front)
                break

            fronts.append(current_front)
            for i in current_front:
                front_indices.add(i)

            # 减少被支配计数
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_count[j] = max(0, domination_count[j] - 1)

        return fronts

    @staticmethod
    def _crowding_distance(objectives, front_indices):
        """计算拥挤距离"""
        n_front = len(front_indices)
        if n_front <= 2:
            return np.full(n_front, np.inf)

        distances = np.zeros(n_front)
        for k in range(objectives.shape[1]):
            # 按第k个目标排序
            sorted_idx = sorted(range(n_front),
                               key=lambda i: objectives[front_indices[i], k])
            obj_range = (objectives[front_indices[sorted_idx[-1]], k] -
                        objectives[front_indices[sorted_idx[0]], k])
            if obj_range < 1e-10:
                continue

            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf

            for i in range(1, n_front - 1):
                distances[sorted_idx[i]] += (
                    (objectives[front_indices[sorted_idx[i + 1]], k] -
                     objectives[front_indices[sorted_idx[i - 1]], k]) / obj_range
                )

        return distances

    def _tournament_selection(self, pop, objectives, fronts, crowding):
        """锦标赛选择: 2元锦标赛"""
        selected = np.zeros((self.pop_size, self.n_var))
        front_rank = np.zeros(self.pop_size, dtype=int)
        for fi, front in enumerate(fronts):
            for idx in front:
                front_rank[idx] = fi

        for i in range(self.pop_size):
            a = self.rng.randint(0, self.pop_size)
            b = self.rng.randint(0, self.pop_size)
            if front_rank[a] < front_rank[b]:
                winner = a
            elif front_rank[b] < front_rank[a]:
                winner = b
            elif crowding.get(a, 0) > crowding.get(b, 0):
                winner = a
            else:
                winner = b
            selected[i] = pop[winner]

        return selected

    def _sbx_crossover(self, parent1, parent2):
        """模拟二进制交叉 (SBX)"""
        if self.rng.random() > self.p_cross:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        for j in range(self.n_var):
            if self.rng.random() > 0.5:
                if abs(parent1[j] - parent2[j]) > 1e-10:
                    if parent1[j] < parent2[j]:
                        y1, y2 = parent1[j], parent2[j]
                    else:
                        y1, y2 = parent2[j], parent1[j]

                    lb, ub = self.bounds[j]
                    beta = 1.0 + 2.0 * min(y1 - lb, ub - y2) / max(y2 - y1, 1e-10)

                    alpha = 2.0 - beta ** -(self.eta_c + 1)
                    u = self.rng.random()
                    if u <= 1.0 / alpha:
                        beta_q = (u * alpha) ** (1.0 / (self.eta_c + 1))
                    else:
                        beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1))

                    child1[j] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    child2[j] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

        return child1, child2

    def _polynomial_mutation(self, individual):
        """多项式变异"""
        mutant = individual.copy()
        for j in range(self.n_var):
            if self.rng.random() < self.p_mut:
                lb, ub = self.bounds[j]
                delta = min(ub - mutant[j], mutant[j] - lb) / (ub - lb)
                u = self.rng.random()
                if u < 0.5:
                    delta_q = (2.0 * u + (1.0 - 2.0 * u) *
                              (1.0 - delta) ** (self.eta_m + 1)) ** (1.0 / (self.eta_m + 1)) - 1.0
                else:
                    delta_q = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) *
                                    (1.0 - delta) ** (self.eta_m + 1)) ** (1.0 / (self.eta_m + 1))
                mutant[j] += delta_q * (ub - lb)

        return mutant

    def optimize(self, verbose=True):
        """运行NSGA-II优化

        Returns
        -------
        pareto_solutions : list[dict]
            Pareto最优解集, 每个元素为 {pv, ess_cap, ess_pow, npc, ssr, carbon, ...}
        pareto_objectives : np.ndarray
            对应的目标值矩阵
        history : list
            每代Pareto前沿大小记录
        """
        # 初始化
        pop = self._initialize()
        objectives, results = self._evaluate_population(pop)

        history = []

        for gen in range(self.n_gen):
            # 非支配排序
            fronts = self._non_dominated_sort(objectives)

            # 拥挤距离
            crowding = {}
            for front in fronts:
                cd = self._crowding_distance(objectives, front)
                for i, idx in enumerate(front):
                    crowding[idx] = cd[i]

            # 锦标赛选择
            selected = self._tournament_selection(pop, objectives, fronts, crowding)

            # 交叉
            offspring = np.zeros_like(pop)
            for i in range(0, self.pop_size, 2):
                child1, child2 = self._sbx_crossover(selected[i], selected[i + 1])
                offspring[i] = child1
                offspring[i + 1] = child2

            # 变异
            for i in range(self.pop_size):
                offspring[i] = self._polynomial_mutation(offspring[i])
                # 边界裁剪
                offspring[i] = np.clip(offspring[i], self.bounds[:, 0], self.bounds[:, 1])
                offspring[i, 2] = min(offspring[i, 2], offspring[i, 1] * 0.5)

            # 合并父代和子代
            combined_pop = np.vstack([pop, offspring])
            combined_obj, combined_results = self._evaluate_population(combined_pop)

            # 精英保留: 选择前pop_size个
            fronts_combined = self._non_dominated_sort(combined_obj)
            crowding_combined = {}
            for front in fronts_combined:
                cd = self._crowding_distance(combined_obj, front)
                for i, idx in enumerate(front):
                    crowding_combined[idx] = cd[i]

            new_pop = np.zeros((self.pop_size, self.n_var))
            new_obj = np.zeros((self.pop_size, self.n_obj))
            new_results = []
            filled = 0

            for front in fronts_combined:
                if filled + len(front) <= self.pop_size:
                    for idx in front:
                        new_pop[filled] = combined_pop[idx]
                        new_obj[filled] = combined_obj[idx]
                        new_results.append(combined_results[idx])
                        filled += 1
                else:
                    # 按拥挤距离截断
                    remaining = self.pop_size - filled
                    sorted_front = sorted(front,
                                         key=lambda idx: crowding_combined.get(idx, 0),
                                         reverse=True)
                    for idx in sorted_front[:remaining]:
                        new_pop[filled] = combined_pop[idx]
                        new_obj[filled] = combined_obj[idx]
                        new_results.append(combined_results[idx])
                        filled += 1
                    break

            pop = new_pop
            objectives = new_obj
            results = new_results

            history.append(len(fronts_combined[0]))

            if verbose and (gen + 1) % 10 == 0:
                f0 = fronts_combined[0]
                best_npc = min(objectives[f_idx, 0] for f_idx in f0)
                best_ssr = 1 - min(objectives[f_idx, 1] for f_idx in f0)
                print(f"  NSGA-II gen {gen+1}/{self.n_gen}: "
                      f"Pareto size={len(f0)}, "
                      f"NPC range=[{objectives[f0,0].min()/1e4:.0f}, "
                      f"{objectives[f0,0].max()/1e4:.0f}]万元, "
                      f"SSR range=[{1-objectives[f0,1].max():.1%}, "
                      f"{1-objectives[f0,1].min():.1%}]")

        # 最终非支配排序
        fronts_final = self._non_dominated_sort(objectives)
        pareto_front = fronts_final[0]

        # 构建Pareto解集
        pareto_solutions = []
        for idx in pareto_front:
            r = results[idx]
            if r is None:
                continue
            pareto_solutions.append({
                'pv_capacity': float(pop[idx, 0]),
                'ess_capacity': float(pop[idx, 1]),
                'ess_power': float(pop[idx, 2]),
                'npc': float(objectives[idx, 0]),
                'self_sufficiency': float(1.0 - objectives[idx, 1]),
                'carbon_reduction_t': float(-objectives[idx, 2]),
                'ssr_pct': float((1.0 - objectives[idx, 1]) * 100),
                'npc_wan_yuan': float(objectives[idx, 0] / 1e4),
                **{k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in r.items()
                   if k not in ['capex_detail', 'verify_8760h']},
            })

        # 按SSR排序
        pareto_solutions.sort(key=lambda x: x['self_sufficiency'])

        pareto_objectives = objectives[pareto_front]

        return pareto_solutions, pareto_objectives, history


def print_pareto_summary(pareto_solutions):
    """打印Pareto前沿摘要"""
    print(f"\n{'='*65}")
    print(f"NSGA-II Pareto最优前沿 ({len(pareto_solutions)} 个非支配解)")
    print(f"{'='*65}")
    print(f"  {'PV(kWp)':>8s} {'ESS(kWh)':>9s} {'ESS(kW)':>8s} "
          f"{'NPC(万元)':>9s} {'SSR':>7s} {'Carbon(t)':>9s}")
    print(f"  {'-'*58}")
    for s in pareto_solutions:
        print(f"  {s['pv_capacity']:>8.0f} {s['ess_capacity']:>9.0f} "
              f"{s['ess_power']:>8.0f} {s['npc_wan_yuan']:>9.1f} "
              f"{s['ssr_pct']:>6.1f}% {s['carbon_reduction_t']:>9.1f}")


def save_pareto_results(pareto_solutions, output_dir='results'):
    """保存Pareto结果到JSON"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'pareto_nsga2_results.json')

    save_list = []
    for s in pareto_solutions:
        save_list.append({
            'pv_capacity': s['pv_capacity'],
            'ess_capacity': s['ess_capacity'],
            'ess_power': s['ess_power'],
            'npc_wan_yuan': s['npc_wan_yuan'],
            'self_sufficiency': s['self_sufficiency'],
            'ssr_pct': s['ssr_pct'],
            'carbon_reduction_t': s['carbon_reduction_t'],
        })

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, indent=2, ensure_ascii=False)
    print(f"Pareto结果已保存: {os.path.abspath(path)}")


if __name__ == '__main__':
    print("NSGA-II 模块加载成功")
    print("此模块需要通过 main.py 调用 (需传入 MicrogridOptimizer)")
    print("用法: from nsga2 import NSGA2, print_pareto_summary, save_pareto_results")
