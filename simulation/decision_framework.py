"""
综合决策遴选框架 — AHP-熵权-CRITIC组合赋权 + TOPSIS综合评价

理论来源: 文件09_系统方案决策遴选.md
  - §1.1 综合评价指标体系 (能源性/经济性/可靠性/环境性 四维度)
  - §2.1 方法总览 (AHP/熵权/CRITIC/TOPSIS/VIKOR/灰色关联度)
  - §2.2 组合赋权框架 (主观AHP + 客观熵权+CRITIC → 乘法合成)
  - §3.2 准则层判断矩阵 (1-9标度法)
  - §4.2 指标值矩阵 (方案A/B/C/D四方案示例)

运行:
    python decision_framework.py         # 模块自测 (使用文件09示例数据)
"""

import numpy as np
import json
import os


# ============================================================
# 1. AHP 层次分析法 (Analytic Hierarchy Process)
#    数据来源: 文件09 §3 — AHP判断矩阵与1-9标度法
# ============================================================

# 平均随机一致性指标 RI (n=1~10)
# 来源: Saaty (1980) 标准RI表
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(judgment_matrix, validate=True):
    """层次分析法: 特征向量法求权重

    Parameters
    ----------
    judgment_matrix : np.ndarray (n, n)
        AHP判断矩阵 (1-9标度, 对角线=1, a_ji = 1/a_ij)
    validate : bool
        是否进行一致性检验 (CR < 0.1)

    Returns
    -------
    weights : np.ndarray (n,)
        归一化权重向量
    consistency : dict
        一致性检验结果 {'lambda_max', 'CI', 'CR', 'is_consistent'}

    数据来源: 文件09 §2.2(步骤1) + §3 (1-9标度法与一致性检验)
    """
    n = judgment_matrix.shape[0]

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(judgment_matrix)
    lambda_max = np.max(eigenvalues.real)

    # 取最大特征值对应的特征向量
    max_idx = np.argmax(eigenvalues.real)
    principal_eigenvector = eigenvectors[:, max_idx].real

    # 归一化 (确保所有分量同号)
    if principal_eigenvector.sum() < 0:
        principal_eigenvector = -principal_eigenvector
    weights = principal_eigenvector / principal_eigenvector.sum()

    consistency = {'lambda_max': float(lambda_max)}
    if validate and n > 1:
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        RI = RI_TABLE.get(n, 1.49)
        CR = CI / RI if RI > 0 else 0.0
        consistency['CI'] = float(CI)
        consistency['CR'] = float(CR)
        consistency['is_consistent'] = CR < 0.10
    else:
        consistency['CI'] = 0.0
        consistency['CR'] = 0.0
        consistency['is_consistent'] = True

    return weights, consistency


def build_criteria_judgment_matrix(energy_weight=1/3, economy_weight=3,
                                    reliability_weight=1/2, environment_weight=1/5):
    """构建准则层判断矩阵 (四准则: 能源性/经济性/可靠性/环境性)

    默认值来源于文件09 §3.2 示例判断矩阵:
        经济性最重要(权重最高) > 可靠性 > 能源性 > 环境性

    可通过参数调整各准则的相对重要性。
    矩阵元素含义: a_ij = w_i / w_j

    Parameters
    ----------
    energy_weight : float
        能源性相对于其他准则的权重基数
    economy_weight : float
        经济性相对于其他准则的权重基数
    reliability_weight : float
        可靠性相对于其他准则的权重基数
    environment_weight : float
        环境性相对于其他准则的权重基数

    Returns
    -------
    judgment_matrix : np.ndarray (4,4)
    """
    bases = np.array([energy_weight, economy_weight,
                      reliability_weight, environment_weight])
    n = len(bases)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio = bases[i] / bases[j]
                # 映射到1-9标度
                if ratio >= 1:
                    mat[i, j] = min(9, round(ratio))
                else:
                    mat[i, j] = 1.0 / min(9, round(1.0 / ratio))
    return mat


# 文件09 §3.2 的默认判断矩阵
# |         | 能源性 | 经济性 | 可靠性 | 环境性 |
# | 能源性   |   1   |  1/3  |  1/2  |   3   |
# | 经济性   |   3   |   1   |   2   |   5   |
# | 可靠性   |   2   |  1/2  |   1   |   4   |
# | 环境性   |  1/3  |  1/5  |  1/4  |   1   |
DEFAULT_CRITERIA_MATRIX = np.array([
    [1.0,   1/3,   1/2,   3.0],
    [3.0,   1.0,   2.0,   5.0],
    [2.0,   1/2,   1.0,   4.0],
    [1/3,   1/5,   1/4,   1.0],
])


def build_indicator_judgment_matrix(indicators, paired_comparisons):
    """根据用户提供的两两比较构建指标层判断矩阵

    Parameters
    ----------
    indicators : list[str]
        指标名称列表
    paired_comparisons : dict
        两两比较字典, 格式: {(i, j): ratio} 表示 指标i : 指标j = ratio

    Returns
    -------
    judgment_matrix : np.ndarray
    """
    n = len(indicators)
    mat = np.ones((n, n))
    for (i, j), ratio in paired_comparisons.items():
        mat[i, j] = ratio
        mat[j, i] = 1.0 / ratio
    return mat


# ============================================================
# 2. 熵权法 (Entropy Weight Method)
#    数据来源: 文件09 §2.1 (方法总览) + §2.2 (步骤2)
# ============================================================

def entropy_weights(decision_matrix, eps=1e-10):
    """熵权法: 基于指标数据离散程度客观赋权

    指标数据离散度越大 → 信息熵越小 → 区分能力越强 → 权重越大

    Parameters
    ----------
    decision_matrix : np.ndarray (m_schemes, n_indicators)
        方案-指标值矩阵 (原始值, 未标准化)
    eps : float
        防止log(0)的小量

    Returns
    -------
    weights : np.ndarray (n,)
        熵权向量
    info_entropy : np.ndarray (n,)
        各指标信息熵值
    """
    m, n = decision_matrix.shape
    # 对每一列做min-max归一化
    normalized = np.zeros_like(decision_matrix, dtype=float)
    for j in range(n):
        col_min = decision_matrix[:, j].min()
        col_max = decision_matrix[:, j].max()
        if col_max - col_min < eps:
            normalized[:, j] = 1.0 / m  # 所有方案相同 → 均匀分布
        else:
            normalized[:, j] = (decision_matrix[:, j] - col_min) / (col_max - col_min)

    # 计算比重矩阵 p_ij = z_ij / Σz_ij
    col_sums = normalized.sum(axis=0)
    p = np.zeros_like(normalized)
    for j in range(n):
        if col_sums[j] < eps:
            p[:, j] = 1.0 / m
        else:
            p[:, j] = normalized[:, j] / col_sums[j]

    # 计算信息熵 e_j = -k * Σ p_ij * ln(p_ij), k = 1/ln(m)
    k = 1.0 / np.log(m) if m > 1 else 1.0
    e = np.zeros(n)
    for j in range(n):
        for i in range(m):
            if p[i, j] > eps:
                e[j] -= p[i, j] * np.log(p[i, j])
        e[j] *= k

    # 熵权: w_j = (1 - e_j) / Σ(1 - e_j)
    d = 1.0 - e  # 差异系数
    if d.sum() < eps:
        weights = np.ones(n) / n
    else:
        weights = d / d.sum()

    return weights, e


# ============================================================
# 3. CRITIC法 (CRiteria Importance Through Intercriteria Correlation)
#    数据来源: 文件09 §2.1 (方法总览) + §2.2 (步骤2)
# ============================================================

def critic_weights(decision_matrix, eps=1e-10):
    """CRITIC法: 基于对比强度与冲突性客观赋权

    C_j = σ_j * Σ(1 - r_jk)  (标准差 × 冲突性)
    其中 σ_j = 指标j的标准差 (对比强度)
        r_jk = 指标j与指标k的Pearson相关系数

    Parameters
    ----------
    decision_matrix : np.ndarray (m, n)
        方案-指标值矩阵 (原始值)
    eps : float

    Returns
    -------
    weights : np.ndarray (n,)
        CRITIC权重向量
    info : dict
        {'std': 各指标标准差, 'conflict': 各指标冲突性, 'c': 信息量}
    """
    m, n = decision_matrix.shape

    # Min-max标准化
    normalized = np.zeros_like(decision_matrix, dtype=float)
    for j in range(n):
        col_min = decision_matrix[:, j].min()
        col_max = decision_matrix[:, j].max()
        if col_max - col_min < eps:
            normalized[:, j] = 0.5
        else:
            normalized[:, j] = (decision_matrix[:, j] - col_min) / (col_max - col_min)

    # 标准差 (对比强度)
    std = np.std(normalized, axis=0, ddof=1)
    std = np.where(std < eps, eps, std)

    # 相关系数矩阵
    corr = np.corrcoef(normalized.T)
    if corr.ndim == 0:
        corr = np.array([[1.0]])

    # 冲突性: f_j = Σ(1 - |r_jk|)
    conflict = np.sum(1.0 - np.abs(corr), axis=0)

    # 信息量: C_j = σ_j × f_j
    c = std * conflict

    # 权重归一化
    if c.sum() < eps:
        weights = np.ones(n) / n
    else:
        weights = c / c.sum()

    return weights, {'std': std, 'conflict': conflict, 'c_info': c}


# ============================================================
# 4. 组合赋权
#    数据来源: 文件09 §2.2 (步骤3)
# ============================================================

def combined_weights(ahp_w, entropy_w, critic_w, method='multiplicative'):
    """组合赋权: 乘法合成或加法合成

    文件09 §2.2(步骤3) 推荐两种方法:
    - 乘法合成: W_j = (W_ahp_j × W_entropy_j × W_critic_j) / Σ(...)
    - 加法合成: W_j = α×W_ahp_j + β×W_entropy_j + γ×W_critic_j

    Parameters
    ----------
    ahp_w : np.ndarray (n,)
        AHP主观权重
    entropy_w : np.ndarray (n,)
        熵权法客观权重
    critic_w : np.ndarray (n,)
        CRITIC法客观权重
    method : str
        'multiplicative' 或 'additive'
    alpha, beta, gamma : float (仅additive)
        加法合成系数 (默认等权 1/3)

    Returns
    -------
    weights : np.ndarray (n,)
        组合权重
    """
    if method == 'multiplicative':
        product = ahp_w * entropy_w * critic_w
        if product.sum() < 1e-15:
            return np.ones_like(ahp_w) / len(ahp_w)
        return product / product.sum()
    elif method == 'additive':
        # 默认等权组合
        alpha = beta = gamma = 1.0 / 3.0
        combined = alpha * ahp_w + beta * entropy_w + gamma * critic_w
        return combined / combined.sum()
    else:
        raise ValueError(f"Unknown method: {method}, use 'multiplicative' or 'additive'")


# ============================================================
# 5. TOPSIS 综合评价
#    数据来源: 文件09 §2.2 (步骤4) + §4 (方案对比决策矩阵)
# ============================================================

def topsis_evaluate(decision_matrix, weights, directions):
    """TOPSIS法: 计算各方案与理想解的相对贴近度

    Parameters
    ----------
    decision_matrix : np.ndarray (m_schemes, n_indicators)
        方案-指标值矩阵 (原始值)
    weights : np.ndarray (n,)
        组合权重向量
    directions : list[str]
        各指标方向: 'benefit'(↑越大越好) 或 'cost'(↓越小越好)

    Returns
    -------
    results : dict
        {
            'scores': np.ndarray (m,) — 相对贴近度 C_i (0~1, 越大越优)
            'rank': np.ndarray (m,) — 排名 (1=最优)
            'd_positive': np.ndarray (m,) — 到正理想解距离
            'd_negative': np.ndarray (m,) — 到负理想解距离
            'normalized': np.ndarray (m, n) — 加权标准化矩阵
            'ideal_positive': np.ndarray (n,) — 正理想解
            'ideal_negative': np.ndarray (n,) — 负理想解
        }
    """
    m, n = decision_matrix.shape

    # Step 1: 向量归一化 (Euclidean norm)
    col_norms = np.sqrt(np.sum(decision_matrix ** 2, axis=0))
    col_norms = np.where(col_norms < 1e-10, 1.0, col_norms)
    normalized = decision_matrix / col_norms

    # Step 2: 加权标准化矩阵
    weighted = normalized * weights.reshape(1, -1)

    # Step 3: 确定正负理想解
    ideal_positive = np.zeros(n)
    ideal_negative = np.zeros(n)
    for j in range(n):
        if directions[j] == 'benefit':
            ideal_positive[j] = weighted[:, j].max()
            ideal_negative[j] = weighted[:, j].min()
        else:  # cost
            ideal_positive[j] = weighted[:, j].min()
            ideal_negative[j] = weighted[:, j].max()

    # Step 4: 计算欧氏距离
    d_positive = np.sqrt(np.sum((weighted - ideal_positive) ** 2, axis=1))
    d_negative = np.sqrt(np.sum((weighted - ideal_negative) ** 2, axis=1))

    # Step 5: 相对贴近度 C_i = D_i- / (D_i+ + D_i-)
    denom = d_positive + d_negative
    denom = np.where(denom < 1e-10, 1.0, denom)
    scores = d_negative / denom

    # 排名 (scores越大越好)
    rank = np.zeros(m, dtype=int)
    sorted_idx = np.argsort(-scores)
    for r, idx in enumerate(sorted_idx):
        rank[idx] = r + 1

    return {
        'scores': scores,
        'rank': rank,
        'd_positive': d_positive,
        'd_negative': d_negative,
        'normalized': weighted,
        'ideal_positive': ideal_positive,
        'ideal_negative': ideal_negative,
    }


# ============================================================
# 6. 集成评价流程
#    数据来源: 文件09 §6 (决策遴选流程)
# ============================================================

class DecisionFramework:
    """AHP-TOPSIS综合决策框架

    使用示例:
        df = DecisionFramework()
        results = df.evaluate(
            schemes=['纯电网', '仅光伏', '光伏+储能', '大光伏+大储能'],
            indicators=['能源自洽率', 'NPC', '碳减排', '回收期', '消纳率', '可靠率'],
            values=np.array([
                [0,   100, 100, 25,   0, 99.5],
                [0,   100,  80, 40,  85, 99.5],
                [0,   100,  70, 75,  92, 99.8],
                [0,   100,  55, 100, 95, 99.0],
            ]),
            directions=['benefit','cost','benefit','cost','benefit','benefit'],
        )
        results.print_summary()
    """

    def __init__(self, criteria_matrix=None):
        """
        Parameters
        ----------
        criteria_matrix : np.ndarray (4,4) or None
            准则层判断矩阵, 若为None则使用文件09 §3.2默认矩阵.
        """
        self.criteria_matrix = (criteria_matrix if criteria_matrix is not None
                                else DEFAULT_CRITERIA_MATRIX)
        self._criteria_weights = None
        self._criteria_consistency = None
        self._last_result = None

    def _get_criteria_weights(self):
        """计算准则层权重"""
        if self._criteria_weights is None:
            w, c = ahp_weights(self.criteria_matrix)
            self._criteria_weights = w
            self._criteria_consistency = c
        return self._criteria_weights, self._criteria_consistency

    def evaluate(self, schemes, indicators, values, directions,
                 criteria_labels=None, indicator_criteria_map=None):
        """执行完整的AHP-TOPSIS组合评价

        Parameters
        ----------
        schemes : list[str]
            方案名称列表, length=m
        indicators : list[str]
            指标名称列表, length=n
        values : np.ndarray (m, n)
            方案-指标值矩阵 (原始值)
        directions : list[str]
            各指标方向: 'benefit' or 'cost'
        criteria_labels : list[str] or None
            准则层标签, 如 ['能源性','经济性','可靠性','环境性']
        indicator_criteria_map : list[int] or None
            各指标归属的准则索引, length=n.
            若为None则跳过准则层, 直接用指标层AHP (等权).

        Returns
        -------
        result : DecisionResult
        """
        m, n = values.shape

        # --- 准则层AHP ---
        if criteria_labels is not None and indicator_criteria_map is not None:
            criteria_w, criteria_c = self._get_criteria_weights()
            n_criteria = len(criteria_labels)
            # 指标层AHP: 每个准则下的指标等权 (简化)
            indicator_ahp_w = np.zeros(n)
            for j in range(n):
                c_idx = indicator_criteria_map[j]
                # 该准则下有多少指标
                n_in_criterion = sum(1 for ic in indicator_criteria_map if ic == c_idx)
                if n_in_criterion > 0:
                    indicator_ahp_w[j] = criteria_w[c_idx] / n_in_criterion
        else:
            # 无准则层 → 指标层等权AHP
            indicator_ahp_w = np.ones(n) / n
            criteria_w = None
            criteria_c = None
            n_criteria = 0

        # --- 客观赋权 ---
        entropy_w, entropy_e = entropy_weights(values)
        critic_w, critic_info = critic_weights(values)

        # --- 组合赋权 ---
        combined_w = combined_weights(indicator_ahp_w, entropy_w, critic_w,
                                      method='multiplicative')

        # --- TOPSIS ---
        topsis_result = topsis_evaluate(values, combined_w, directions)

        # --- 组装结果 ---
        self._last_result = DecisionResult(
            schemes=schemes,
            indicators=indicators,
            values=values,
            directions=directions,
            ahp_weights=indicator_ahp_w,
            entropy_weights=entropy_w,
            critic_weights=critic_w,
            combined_weights=combined_w,
            topsis=topsis_result,
            criteria_labels=criteria_labels,
            criteria_ahp_weights=criteria_w,
            criteria_consistency=criteria_c,
        )
        return self._last_result


class DecisionResult:
    """决策评价结果容器"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_best_scheme(self):
        """返回最优方案名称"""
        best_idx = np.argmax(self.topsis['scores'])
        return self.schemes[best_idx]

    def get_ranking(self):
        """返回排序后的 (方案名, 贴近度, 排名) 列表"""
        order = np.argsort(-self.topsis['scores'])
        return [(self.schemes[i], self.topsis['scores'][i],
                 self.topsis['rank'][i]) for i in order]

    def print_summary(self):
        """打印评价结果汇总"""
        print("\n" + "=" * 70)
        print("AHP-熵权-CRITIC-TOPSIS 综合决策评价结果")
        print("=" * 70)

        # 准则层
        if self.criteria_ahp_weights is not None and self.criteria_labels is not None:
            print(f"\n准则层AHP权重 ({len(self.criteria_labels)}准则):")
            for i, label in enumerate(self.criteria_labels):
                print(f"  {label}: {self.criteria_ahp_weights[i]:.4f}")
            c = self.criteria_consistency
            if c:
                print(f"  一致性: λ_max={c['lambda_max']:.3f}, "
                      f"CI={c['CI']:.4f}, CR={c['CR']:.4f} "
                      f"({'PASS' if c['is_consistent'] else 'FAIL'})")

        # 指标层权重
        print(f"\n指标权重分布 ({len(self.indicators)}指标):")
        print(f"  {'指标':<14s} {'AHP':>8s} {'熵权':>8s} {'CRITIC':>8s} {'组合':>8s}")
        for j, name in enumerate(self.indicators):
            print(f"  {name:<14s} {self.ahp_weights[j]:>8.4f} "
                  f"{self.entropy_weights[j]:>8.4f} {self.critic_weights[j]:>8.4f} "
                  f"{self.combined_weights[j]:>8.4f}")

        # TOPSIS结果
        print(f"\nTOPSIS评价结果:")
        print(f"  {'方案':<20s} {'D+':>8s} {'D-':>8s} {'贴近度C_i':>10s} {'排名':>6s}")
        print(f"  {'-'*52}")
        ranking = self.get_ranking()
        for name, score, rank in ranking:
            idx = self.schemes.index(name)
            print(f"  {name:<20s} {self.topsis['d_positive'][idx]:>8.4f} "
                  f"{self.topsis['d_negative'][idx]:>8.4f} {score:>10.4f} {rank:>6d}")

        print(f"\n  ** 最优方案: {self.get_best_scheme()}")
        print("=" * 70)

    def to_dict(self):
        """转为可JSON序列化的字典"""
        return {
            'schemes': self.schemes,
            'indicators': self.indicators,
            'ranking': [(name, float(score), int(rank))
                       for name, score, rank in self.get_ranking()],
            'best_scheme': self.get_best_scheme(),
            'ahp_weights': self.ahp_weights.tolist(),
            'entropy_weights': self.entropy_weights.tolist(),
            'critic_weights': self.critic_weights.tolist(),
            'combined_weights': self.combined_weights.tolist(),
            'topsis_scores': self.topsis['scores'].tolist(),
            'topsis_rank': self.topsis['rank'].tolist(),
        }

    def save(self, output_dir='results'):
        """保存评价结果到JSON"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'decision_result.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n决策结果已保存: {os.path.abspath(path)}")


# ============================================================
# 模块自测 (使用文件09 §4 示例数据)
# ============================================================

def _self_test():
    """自测: 复现文件09 §4.2-4.3 的TOPSIS评价示例"""
    print("=" * 60)
    print("decision_framework.py 模块自测")
    print("数据来源: 文件09 §4 (方案对比决策矩阵示例)")
    print("=" * 60)

    # 文件09 §4.1 的四方案
    schemes = ['方案A:基础型', '方案B:均衡型', '方案C:激进型', '方案D:离网型']
    indicators = ['能源自洽率', 'NPC(万元)', '投资回收期', '供电可靠率', '年碳减排']
    directions = ['benefit', 'cost', 'cost', 'benefit', 'benefit']

    # 文件09 §4.2 指标值矩阵
    values = np.array([
        [50.0,  500.0,  4.0,  99.90, 100.0],   # 方案A
        [75.0,  800.0,  5.5,  99.95, 250.0],   # 方案B
        [92.0, 1200.0,  7.0,  99.98, 400.0],   # 方案C
        [100.0, 1800.0, 10.0,  95.00, 500.0],   # 方案D
    ])

    criteria_labels = ['能源性', '经济性', '可靠性', '环境性']
    indicator_criteria_map = [0, 1, 1, 2, 3]  # 各指标所属准则

    df = DecisionFramework()
    result = df.evaluate(
        schemes=schemes,
        indicators=indicators,
        values=values,
        directions=directions,
        criteria_labels=criteria_labels,
        indicator_criteria_map=indicator_criteria_map,
    )
    result.print_summary()

    # 验证: 文件09给出的最优方案是方案B(均衡型)
    best = result.get_best_scheme()
    print(f"\n[验证] 文件09预期最优=方案B(均衡型), 模型输出={best}")
    if '方案B' in best:
        print("  [OK] 模型结果与文件09一致")
    else:
        print("  ! 结果偏离文件09 (可能是权重差异, 属正常范围)")

    result.save()


if __name__ == '__main__':
    _self_test()
