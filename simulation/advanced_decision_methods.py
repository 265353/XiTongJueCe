"""
高级决策方法模块 — VIKOR折中排序 + 灰色关联度分析(GRA) + 权重敏感性分析

理论来源: 文件09_系统方案决策遴选.md
  - §2.1 方法总览 (VIKOR/灰色关联度)
  - §5.2 敏感性分析

补充现有 decision_framework.py 的 AHP-TOPSIS, 提供多方法交叉验证.

运行:
    python advanced_decision_methods.py     # 模块自测
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# 1. VIKOR 折中排序法 (VlseKriterijumska Optimizacija I Kompromisno Resenje)
#    原理: 最大化群体效用 + 最小化个体遗憾 → 折中最优
#    来源: 文件09 §2.1
# ============================================================

def vikor(values, weights, directions, v=0.5):
    """VIKOR多准则折中排序

    Parameters
    ----------
    values : np.ndarray (m, n)
        决策矩阵, m个方案, n个指标
    weights : np.ndarray (n,)
        指标权重 (归一化)
    directions : list of str
        指标方向: 'benefit'(越大越好) 或 'cost'(越小越好)
    v : float
        群体效用权重 (默认0.5, 折中态度)
        v>0.5 → 偏重群体效用 (多数决)
        v<0.5 → 偏重个体遗憾 (否决权)

    Returns
    -------
    result : dict
        S_i (群体效用), R_i (个体遗憾), Q_i (折中排序指标)
        ranking (按Q_i升序, Q_i越小越优)
        compromise_solution (是否满足可接受优势/稳定性条件)
    """
    m, n = values.shape
    directions_arr = np.array([1 if d == 'benefit' else -1 for d in directions])

    # Step 1: 确定正理想解 f* 和负理想解 f-
    f_star = np.zeros(n)
    f_minus = np.zeros(n)
    for j in range(n):
        if directions_arr[j] == 1:
            f_star[j] = values[:, j].max()
            f_minus[j] = values[:, j].min()
        else:
            f_star[j] = values[:, j].min()
            f_minus[j] = values[:, j].max()

    # Step 2: 计算 S_i (群体效用) 和 R_i (个体遗憾)
    S = np.zeros(m)
    R = np.zeros(m)
    for i in range(m):
        for j in range(n):
            if f_star[j] != f_minus[j]:
                d_ij = abs(f_star[j] - values[i, j]) / abs(f_star[j] - f_minus[j])
            else:
                d_ij = 0
            S[i] += weights[j] * d_ij
            R[i] = max(R[i], weights[j] * d_ij)

    # Step 3: 计算 Q_i (折中排序指标)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = np.zeros(m)
    for i in range(m):
        if S_minus != S_star:
            Q[i] += v * (S[i] - S_star) / (S_minus - S_star)
        if R_minus != R_star:
            Q[i] += (1 - v) * (R[i] - R_star) / (R_minus - R_star)

    # Step 4: 排序 (Q越小越优)
    ranking = np.argsort(Q)

    # Step 5: 可接受条件检验
    # 条件1: 可接受优势 Q(a'') - Q(a') >= 1/(m-1)
    condition1 = (Q[ranking[1]] - Q[ranking[0]]) >= 1.0 / (m - 1)

    # 条件2: 可接受稳定性 (a'在S或R中也排第一)
    condition2 = (np.argmin(S) == ranking[0]) or (np.argmin(R) == ranking[0])

    compromise = condition1 and condition2
    if not compromise:
        solution_set = [ranking[0]]
        if not condition1:
            for k in range(1, m):
                if (Q[ranking[k]] - Q[ranking[0]]) < 1.0 / (m - 1):
                    solution_set.append(ranking[k])
                else:
                    break
    else:
        solution_set = [ranking[0]]

    return {
        'S': S, 'R': R, 'Q': Q,
        'ranking': ranking,
        'compromise_solution': compromise,
        'solution_set': solution_set,
        'f_star': f_star, 'f_minus': f_minus,
        'S_star': S_star, 'S_minus': S_minus,
        'R_star': R_star, 'R_minus': R_minus,
    }


# ============================================================
# 2. 灰色关联度分析 (Grey Relational Analysis, GRA)
#    原理: 基于曲线几何相似性度量方案与理想方案的关联度
#    来源: 文件09 §2.1 (小样本/信息不完全场景)
# ============================================================

def grey_relational_analysis(values, directions, rho=0.5):
    """灰色关联度分析

    Parameters
    ----------
    values : np.ndarray (m, n)
        决策矩阵
    directions : list of str
        'benefit' / 'cost'
    rho : float
        分辨系数 (0<rho≤1, 通常0.5)

    Returns
    -------
    result : dict
        grey_relational_degree (关联度, 越大越优)
        ranking (降序)
    """
    m, n = values.shape
    directions_arr = np.array([1 if d == 'benefit' else -1 for d in directions])

    # Step 1: 无量纲化 (极差归一化)
    normalized = np.zeros_like(values, dtype=float)
    for j in range(n):
        vj = values[:, j]
        vmin, vmax = vj.min(), vj.max()
        if vmax != vmin:
            if directions_arr[j] == 1:
                normalized[:, j] = (vj - vmin) / (vmax - vmin)
            else:
                normalized[:, j] = (vmax - vj) / (vmax - vmin)
        else:
            normalized[:, j] = 1.0

    # Step 2: 确定参考序列 (理想方案 = 全1)
    reference = np.ones(n)

    # Step 3: 计算关联系数
    abs_diff = np.abs(reference - normalized)
    min_diff = abs_diff.min()
    max_diff = abs_diff.max()

    xi = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            xi[i, j] = (min_diff + rho * max_diff) / \
                       (abs_diff[i, j] + rho * max_diff)

    # Step 4: 计算关联度 (等权平均)
    grey_degree = xi.mean(axis=1)

    ranking = np.argsort(-grey_degree)  # 降序

    return {
        'normalized': normalized,
        'reference': reference,
        'coefficient_matrix': xi,
        'grey_degree': grey_degree,
        'ranking': ranking,
    }


# ============================================================
# 3. 权重敏感性分析
#    来源: 文件09 §5.2 (分析权重变化对方案排序的影响)
# ============================================================

def weight_sensitivity_analysis(full_evaluator, base_weights, criteria_names,
                                 perturb_range=(-0.3, 0.3), steps=10):
    """权重敏感性分析 — 逐个指标扰动权重, 观察排序稳定性

    Parameters
    ----------
    full_evaluator : callable
        签名为 evaluator(weights) -> ranking_array
    base_weights : np.ndarray (n,)
        基准权重
    criteria_names : list of str
        指标名称
    perturb_range : tuple
        扰动范围 (相对变化)
    steps : int
        扰动步数

    Returns
    -------
    result : dict
        每个指标的排序变化统计
    """
    n = len(base_weights)
    results = {}

    for j in range(n):
        perturbations = np.linspace(perturb_range[0], perturb_range[1], steps)
        ranking_changes = []
        stable_count = 0
        base_ranking = full_evaluator(base_weights)

        for delta in perturbations:
            w = base_weights.copy()
            w[j] = base_weights[j] * (1 + delta)
            # 重新归一化
            w = w / w.sum()
            ranking = full_evaluator(w)
            # 方案排序变化 (Kendall tau距离的近似)
            rank_diff = np.sum(np.abs(np.array(ranking) - np.array(base_ranking)))
            ranking_changes.append(rank_diff)
            if rank_diff == 0:
                stable_count += 1

        results[criteria_names[j]] = {
            'perturbations': perturbations.tolist(),
            'ranking_changes': ranking_changes,
            'stability': stable_count / steps,
            'max_change': max(ranking_changes) if ranking_changes else 0,
        }

    return results


# ============================================================
# 4. 综合决策报告 (多方法交叉验证)
# ============================================================

class MultiMethodDecision:
    """多方法决策器 — TOPSIS + VIKOR + GRA 交叉验证

    使用方法:
        mdm = MultiMethodDecision(schemes, indicators, values, directions)
        mdm.run_all()
        mdm.print_comparison()
    """

    def __init__(self, schemes, indicators, values, directions,
                 weights=None, criteria_labels=None):
        self.schemes = list(schemes)
        self.indicators = list(indicators)
        self.values = np.array(values, dtype=float)
        self.directions = list(directions)
        self.n_schemes = len(schemes)
        self.n_indicators = len(indicators)

        if weights is None:
            weights = np.ones(self.n_indicators) / self.n_indicators
        self.weights = np.array(weights, dtype=float)
        self.criteria_labels = criteria_labels

    def run_all(self):
        """执行所有三种方法"""
        # VIKOR
        self.vikor_result = vikor(self.values, self.weights, self.directions)

        # GRA
        self.gra_result = grey_relational_analysis(self.values, self.directions)

        # 加权得分 (简化TOPSIS近似)
        self.weighted_scores = self._weighted_score()

        return self

    def _weighted_score(self):
        """归一化加权得分"""
        scored = np.zeros((self.n_schemes, self.n_indicators))
        for j in range(self.n_indicators):
            vj = self.values[:, j]
            vmin, vmax = vj.min(), vj.max()
            if vmax != vmin:
                if self.directions[j] == 'benefit':
                    scored[:, j] = (vj - vmin) / (vmax - vmin)
                else:
                    scored[:, j] = (vmax - vj) / (vmax - vmin)
            else:
                scored[:, j] = 1.0
        return (scored * self.weights).sum(axis=1)

    def print_comparison(self):
        """打印三种方法交叉对比"""
        print("\n" + "=" * 65)
        print("多方法决策交叉验证 — VIKOR / GRA / TOPSIS")
        print("=" * 65)

        print(f"\n--- 方案 ---")
        for i, s in enumerate(self.schemes):
            print(f"  [{i}] {s}")

        print(f"\n--- VIKOR 折中排序 (v=0.5) ---")
        vr = self.vikor_result
        print(f"  方案  {'S_i':>8} {'R_i':>8} {'Q_i':>8} {'排名':>4}")
        for i in vr['ranking']:
            marker = " ←最优" if i == vr['ranking'][0] else ""
            print(f"  [{i}]   {vr['S'][i]:>6.4f} {vr['R'][i]:>6.4f} "
                  f"{vr['Q'][i]:>6.4f}  {list(vr['ranking']).index(i)+1:>4}{marker}")
        print(f"  可接受折中解: {'是' if vr['compromise_solution'] else '否'} "
              f"(解集: {vr['solution_set']})")

        print(f"\n--- 灰色关联度分析 (ρ=0.5) ---")
        gr = self.gra_result
        for i in gr['ranking']:
            marker = " ←最优" if i == gr['ranking'][0] else ""
            print(f"  [{i}] {self.schemes[i]:<20s} 关联度={gr['grey_degree'][i]:.4f}{marker}")

        print(f"\n--- 加权综合得分 ---")
        ws_rank = np.argsort(-self.weighted_scores)
        for i in ws_rank:
            marker = " ←最优" if i == ws_rank[0] else ""
            print(f"  [{i}] {self.schemes[i]:<20s} 得分={self.weighted_scores[i]:.4f}{marker}")

        # 方法一致率
        self._print_concordance()

    def _print_concordance(self):
        """方法一致性检验"""
        vr_top = self.vikor_result['ranking'][0]
        gr_top = self.gra_result['ranking'][0]
        ws_top = int(np.argmax(self.weighted_scores))

        print(f"\n--- 方法一致性检验 ---")
        print(f"  VIKOR最优: [{vr_top}] {self.schemes[vr_top]}")
        print(f"  GRA最优:   [{gr_top}] {self.schemes[gr_top]}")
        print(f"  加权最优:  [{ws_top}] {self.schemes[ws_top]}")

        if vr_top == gr_top == ws_top:
            print(f"  [OK] 三种方法一致 -> 高置信度推荐")
        elif vr_top == gr_top or vr_top == ws_top or gr_top == ws_top:
            print(f"  [~] 两种方法一致 -> 中等置信度推荐")
        else:
            print(f"  [!!] 三种方法分歧 -> 需进一步分析")

    def get_best_by_consensus(self):
        """多方法投票确定最优方案 (Borda计数)"""
        n = self.n_schemes
        borda = np.zeros(n)
        # VIKOR ranking -> Borda
        for rank, idx in enumerate(self.vikor_result['ranking']):
            borda[idx] += n - rank
        # GRA ranking -> Borda
        for rank, idx in enumerate(self.gra_result['ranking']):
            borda[idx] += n - rank
        # Weighted
        for rank, idx in enumerate(np.argsort(-self.weighted_scores)):
            borda[idx] += n - rank
        return int(np.argmax(borda)), borda


# ============================================================
# 自测
# ============================================================

def self_test():
    """使用文件09的示例数据测试"""
    # 四方案六指标 (文件09示例)
    schemes = ['方案A:纯电网', '方案B:仅光伏', '方案C:光储均衡', '方案D:大光储']
    indicators = ['能源自洽率(%)', 'NPC(万元)', '碳减排(tCO2/年)',
                  '投资回收期(年)', '光伏消纳率(%)', '供电可靠率(%)']
    directions = ['benefit', 'cost', 'benefit', 'cost', 'benefit', 'benefit']

    # 示例数据 (与main.py中方案值一致)
    values = np.array([
        [0.0,  4200,   0,   20.0,   0,  99.5],
        [20.0, 3200, 150,   12.0,  85,  99.5],
        [28.0, 3600, 320,    9.0,  92,  99.8],
        [45.0, 4800, 520,   11.0,  78,  99.0],
    ])

    # 等权测试
    weights = np.ones(6) / 6

    mdm = MultiMethodDecision(schemes, indicators, values, directions, weights)
    mdm.run_all()
    mdm.print_comparison()

    # 敏感性分析
    best_idx, borda = mdm.get_best_by_consensus()
    print(f"\n--- Borda共识投票 ---")
    for i, s in enumerate(schemes):
        print(f"  [{i}] {s}: Borda={borda[i]:.0f}")
    print(f"  共识最优: [{best_idx}] {schemes[best_idx]}")


if __name__ == '__main__':
    self_test()
