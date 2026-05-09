"""
测量误差与数据处理动态演示系统
=============================

面向高校电子测量技术教学的交互式 Dashboard。
基于 Streamlit + Plotly 构建。

功能模块：
- 数据生成器（随机误差、系统误差、粗大误差）
- 误差识别与剔除（3σ、格拉布斯准则）
- 系统误差发现（马利科夫、阿贝-赫梅特判据）
- 随机误差与不确定度分析
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import re
from streamlit import query_params as st_query_params

# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="测量误差与数据处理动态演示系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .edu-box {
        background: linear-gradient(135deg, #3B82F622 0%, #1D4ED822 100%);
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-highlight {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 辅助函数
# =============================================================================

def render_teaching(title: str, content: str, formula: str = None):
    """渲染理论模块"""
    st.markdown(f"""
    <div class="edu-box">
        <h4>{title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)
    if formula:
        st.latex(formula)


def render_result_box(text: str):
    """渲染结果高亮框"""
    st.markdown(f'<div class="result-highlight">{text}</div>', unsafe_allow_html=True)


def iterative_gross_error_removal(data, method):
    """
    迭代剔除粗大误差：每次剔除残差最大的点，直到没有粗大误差
    返回：(最终有效数据索引, 剔除历史)
    """
    current_data = data.copy()
    # original_indices 始终记录 current_data 中每个元素对应的原始数据索引
    original_indices = np.arange(len(data))
    removed_history = []

    while len(current_data) >= 3:
        n = len(current_data)
        x_bar = np.mean(current_data)
        residuals = current_data - x_bar

        if method == "莱特准则 (3σ)":
            s = np.sqrt(np.sum(residuals**2) / (n - 1))
            threshold = 3 * s
            is_gross_mask = np.abs(residuals) > threshold
        else:  # 格拉布斯准则
            if n < 3:
                break
            s = np.sqrt(np.sum(residuals**2) / (n - 1))
            if s == 0:
                break
            t_crit = stats.t.ppf(0.975, n - 2)
            G_crit = (n - 1) / np.sqrt(n) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
            threshold = G_crit * s
            is_gross_mask = np.abs(residuals) > threshold

        # 找出残差最大的点
        max_residual_idx = np.argmax(np.abs(residuals))

        if is_gross_mask[max_residual_idx]:
            removed_history.append({
                'original_index': original_indices[max_residual_idx],
                'value': current_data[max_residual_idx],
                'residual': residuals[max_residual_idx],
                'threshold': threshold,
                'sigma': s if method == "莱特准则 (3σ)" else G_crit
            })
            # 删除时同步删除对应的原始索引
            current_data = np.delete(current_data, max_residual_idx)
            original_indices = np.delete(original_indices, max_residual_idx)
        else:
            break

    # 构建原始数据的有效掩码
    valid_mask = np.ones(len(data), dtype=bool)
    for removed in removed_history:
        valid_mask[removed['original_index']] = False

    return valid_mask, removed_history


def parse_manual_data(text: str) -> tuple:
    """
    解析手动输入的数据
    支持逗号、空格、换行分隔
    """
    # 替换各种分隔符为空格
    text = re.sub(r'[,;\n\t]+', ' ', text)
    # 分割并转换
    parts = text.split()
    values = []
    for p in parts:
        try:
            values.append(float(p.strip()))
        except ValueError:
            return None, f"无法解析 '{p}'，请输入有效数字"
    if len(values) < 3:
        return None, "数据量不足，至少需要3个数据点"
    return np.array(values), None


# =============================================================================
# 侧边栏
# =============================================================================

with st.sidebar:
    st.markdown("## 数据参数设置")

    # 教学模式开关
    teaching_mode = st.toggle("教学模式", value=True)

    st.markdown("---")

    # 数据源选择
    data_mode = st.radio(
        "数据来源",
        ["自动模拟生成", "手动数据录入"],
        index=0
    )

    if data_mode == "自动模拟生成":
        st.markdown("### 自动生成参数")

        # 真值设定
        A0 = st.number_input("真值 A₀", value=50.0, min_value=0.0, max_value=1000.0, step=1.0)

        # 样本量
        N = st.slider("样本数量 N", 5, 100, 20)

        # 随机误差
        st.markdown("**随机误差 (σ)**")
        sigma = st.slider("σ 标准差", 0.1, 10.0, 1.0, 0.1, key="sigma_init")

        # 仅随机误差数据生成按钮
        generate_btn = st.button("生成模拟数据", type="primary", use_container_width=True)

        if generate_btn:
            rng = np.random.default_rng()
            baseline = np.full(N, A0)
            random_err = rng.normal(0, sigma, N)
            raw_data = baseline + random_err

            st.session_state['raw_data'] = raw_data.copy()
            st.session_state['base_random_data'] = raw_data.copy()
            st.session_state['baseline'] = baseline
            st.session_state['cleaned_data'] = raw_data.copy()
            st.session_state['removed_mask'] = np.ones(N, dtype=bool)
            st.session_state['has_systematic'] = False
            st.session_state['has_gross'] = False
            st.session_state['confirmed_removed_values'] = []
            st.session_state['params'] = {
                'A0': A0, 'N': N, 'sigma': sigma
            }
            st.rerun()

        # 系统误差注入
        st.markdown("---")
        st.markdown("**系统误差注入**")

        C_inj = st.slider("恒值系统误差 C", -10.0, 10.0, 0.0, 0.1, key="C_inj")
        K_inj = st.slider("累进系统误差 K (线性漂移)", -1.0, 1.0, 0.0, 0.01, key="K_inj")
        A_sys_inj = st.slider("周期系统误差幅值 A", 0.0, 10.0, 0.0, 0.1, key="A_sys_inj")
        T_inj = st.slider("周期系统误差周期 T", 1.0, 20.0, 10.0, 0.5, key="T_inj")

        col_sys1, col_sys2 = st.columns(2)
        with col_sys1:
            inject_sys_btn = st.button("注入系统误差", use_container_width=True)
        with col_sys2:
            remove_sys_btn = st.button("清除系统误差", use_container_width=True)

        if inject_sys_btn and 'raw_data' in st.session_state:
            i = np.arange(1, N + 1)
            systematic_err = C_inj + K_inj * i + A_sys_inj * np.sin(2 * np.pi * i / T_inj)
            new_data = st.session_state['base_random_data'] + systematic_err
            st.session_state['raw_data'] = new_data
            st.session_state['cleaned_data'] = new_data.copy()
            st.session_state['removed_mask'] = np.ones(N, dtype=bool)
            st.session_state['has_systematic'] = True
            st.session_state['systematic_params'] = {
                'C': C_inj, 'K': K_inj, 'A_sys': A_sys_inj, 'T': T_inj
            }
            st.rerun()

        if remove_sys_btn and 'raw_data' in st.session_state:
            base = st.session_state['base_random_data'].copy()
            st.session_state['raw_data'] = base
            st.session_state['cleaned_data'] = base.copy()
            st.session_state['removed_mask'] = np.ones(N, dtype=bool)
            st.session_state['has_systematic'] = False
            st.rerun()

        # 粗大误差注入
        st.markdown("**粗大误差注入**")
        gross_count = st.slider("突变点个数", 0, 5, 0, key="gross_count")
        gross_magnitude = st.slider("偏离倍数 (σ)", 3.0, 10.0, 5.0, 0.5, key="gross_magnitude")

        col_gross1, col_gross2 = st.columns(2)
        with col_gross1:
            inject_gross_btn = st.button("注入粗大误差", use_container_width=True)
        with col_gross2:
            remove_gross_btn = st.button("清除粗大误差", use_container_width=True)

        if inject_gross_btn and 'raw_data' in st.session_state:
            st.session_state['gross_method'] = "莱特准则 (3σ)"
            rng = np.random.default_rng()
            N_base = len(st.session_state['base_random_data'])

            if gross_count > 0:
                # 追加新的粗大误差数据点
                gross_new = np.full(gross_count, A0)
                gross_random_err = rng.normal(0, sigma, gross_count)

                # 系统误差索引从基准数据末尾开始排
                i_gross = np.arange(N_base + 1, N_base + gross_count + 1)

                if st.session_state.get('has_systematic'):
                    sys_p = st.session_state.get('systematic_params', {'C': 0, 'K': 0, 'A_sys': 0, 'T': 10})
                    gross_sys_err = sys_p['C'] + sys_p['K'] * i_gross + sys_p['A_sys'] * np.sin(2 * np.pi * i_gross / sys_p['T'])
                    gross_new = gross_new + gross_random_err + gross_sys_err
                else:
                    gross_new = gross_new + gross_random_err

                # 注入粗大偏离
                signs = rng.choice([-1, 1], gross_count)
                gross_new += signs * gross_magnitude * sigma

                # 拼到当前数据后面
                new_data = np.concatenate([st.session_state['raw_data'], gross_new])
            else:
                # gross_count == 0 时，不追加，只标记已注入（空操作）
                new_data = st.session_state['raw_data']

            st.session_state['raw_data'] = new_data
            # removed_mask 扩展（新增点标记为 True）
            old_mask = st.session_state.get('removed_mask', np.ones(len(st.session_state['raw_data']), dtype=bool))
            new_mask = np.concatenate([old_mask, np.ones(gross_count, dtype=bool)]) if gross_count > 0 else old_mask
            st.session_state['removed_mask'] = new_mask
            # cleaned_data 从完整的 x_i（raw_data）按 removed_mask 重新计算
            st.session_state['cleaned_data'] = st.session_state['raw_data'][new_mask]
            st.session_state['has_gross'] = True
            st.session_state['gross_params'] = {
                'count': gross_count, 'magnitude': gross_magnitude
            }
            st.rerun()

        if remove_gross_btn and 'raw_data' in st.session_state:
            N_base = len(st.session_state['base_random_data'])
            base = st.session_state['base_random_data'].copy()
            if st.session_state.get('has_systematic'):
                i = np.arange(1, N_base + 1)
                sys_p = st.session_state.get('systematic_params', {'C': 0, 'K': 0, 'A_sys': 0, 'T': 10})
                systematic_err = sys_p['C'] + sys_p['K'] * i + sys_p['A_sys'] * np.sin(2 * np.pi * i / sys_p['T'])
                base = base + systematic_err
            st.session_state['raw_data'] = base
            st.session_state['cleaned_data'] = base.copy()
            st.session_state['removed_mask'] = np.ones(len(base), dtype=bool)
            st.session_state['confirmed_removed_values'] = []
            st.session_state['has_gross'] = False
            st.rerun()

    else:
        st.markdown("### 手动数据录入")

        data_input = st.text_area(
            "输入测量数据",
            placeholder="用逗号、空格或换行分隔，如: 49.2, 50.1, 48.8, 51.0, 49.5",
            height=120
        )

        if st.button("解析并载入", type="primary", use_container_width=True):
            if data_input.strip():
                values, error = parse_manual_data(data_input)
                if error:
                    st.error(error)
                else:
                    st.session_state['raw_data'] = values
                    st.session_state['baseline'] = np.full(len(values), np.mean(values))
                    st.session_state['cleaned_data'] = values.copy()
                    st.session_state['removed_mask'] = np.ones(len(values), dtype=bool)
                    st.session_state['params'] = {
                        'A0': np.mean(values), 'N': len(values),
                        'sigma': np.std(values, ddof=1)
                    }
                    st.success(f"成功载入 {len(values)} 个数据点")
                    st.rerun()
            else:
                st.warning("请输入数据")

    st.markdown("---")

    # 状态显示
    if 'raw_data' in st.session_state:
        N_valid = np.sum(st.session_state['removed_mask'])
        st.info(f"原始: {len(st.session_state['raw_data'])} | 有效: {N_valid}")


# =============================================================================
# 主界面
# =============================================================================

st.markdown('<h1 class="main-header"><img src="https://raw.githubusercontent.com/jw-zhang0914/measurement-error-system/main/水印.jpg" width="50" style="vertical-align: middle;"> 测量误差与数据处理动态演示系统</h1>', unsafe_allow_html=True)

# 检查数据是否存在
if 'raw_data' not in st.session_state:
    st.info("请在左侧选择数据来源，生成或输入测量数据后开始分析")
    st.stop()

# 获取数据
x_i = st.session_state['raw_data']
clean_mask = st.session_state['removed_mask']
x_valid = st.session_state['cleaned_data']

# 计算统计量（基于有效数据）
N = len(x_i)
N_valid = len(x_valid)
x_bar = np.mean(x_valid)  # 基于有效数据的均值
x_bar_valid = x_bar

# 残差计算（基于有效数据）
v_i = x_i - x_bar  # 用有效数据均值计算全数据残差
v_i_valid = x_valid - x_bar_valid  # 有效数据残差

# 贝塞尔公式计算标准差（基于有效数据）
if N_valid > 1:
    sigma_bessel = np.sqrt(np.sum(v_i_valid**2) / (N_valid - 1))
else:
    sigma_bessel = 0

sigma_bessel_valid = sigma_bessel


# =============================================================================
# Tab 布局
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["数据与残差", "系统误差检验", "粗大误差剔除", "随机误差分析"])


# -----------------------------------------------------------------------------
# Tab 1: 数据与残差
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("## 测量数据与残差分析")

    if teaching_mode:
        st.markdown("""
        <div class="edu-box">
            <h4>算术平均值与残差定义</h4>
            <p>在大量重复测量中，算术平均值是真值的最佳估计值。</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i \quad \text{（算术平均值是真值的最佳估计）}")
        st.latex(r"v_i = x_i - \bar{x} \quad \text{（残差定义）}")
        st.latex(r"\sum_{i=1}^{n} v_i = 0 \quad \text{（残差的抵偿性）}")
        st.caption("注：实验中用贝塞尔公式计算的标准差 s 是总体标准差 σ 的无偏估计，两者意义相同")

    # 指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("样本均值 x̄", f"{x_bar:.4f}")
    with col2:
        st.metric("最大值", f"{np.max(x_i):.4f}")
    with col3:
        st.metric("最小值", f"{np.min(x_i):.4f}")
    with col4:
        st.metric("极差 R", f"{np.max(x_i) - np.min(x_i):.4f}")
    with col5:
        st.metric("σ (贝塞尔)", f"{sigma_bessel:.4f}")

    st.markdown("---")

    indices = np.arange(1, N + 1)

    # 可视化 - 测量数据序列
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=indices, y=x_i,
        mode='lines+markers',
        name='测量值 xᵢ',
        line=dict(color='#4A90D9', width=2),
        marker=dict(size=8),
        hovertemplate='序号: %{x}<br>值: %{y:.4f}'
    ))
    fig1.add_hline(y=x_bar, line_dash='dash', line_color='red',
                   annotation_text=f'x̄ = {x_bar:.2f}')
    fig1.update_layout(
        template='plotly_white',
        height=300,
        title='测量数据序列',
        xaxis_title='测量序号 i',
        yaxis_title='测量值',
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 可视化 - 残差序列
    fig2 = go.Figure()
    colors = ['#EF4444' if not clean_mask[i] else '#4A90D9' for i in range(N)]
    fig2.add_trace(go.Scatter(
        x=indices, y=v_i,
        mode='markers',
        name='残差 vᵢ',
        marker=dict(color=colors, size=10),
        hovertemplate='序号: %{x}<br>残差: %{y:.4f}'
    ))
    fig2.add_hline(y=0, line_dash='dash', line_color='gray')
    fig2.update_layout(
        template='plotly_white',
        height=300,
        title='残差序列',
        xaxis_title='测量序号 i',
        yaxis_title='残差 vᵢ',
        showlegend=False,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 2: 系统误差判别
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("## 系统误差判别")

    if teaching_mode:
        st.markdown("""
        <div class="edu-box">
            <h4>系统误差特点</h4>
            <p>系统误差分为累进性系统误差（线性趋势）和周期性系统误差（周期波动）。它们不会因测量次数增加而减小，需用特定判据检测。</p>
        </div>
        """, unsafe_allow_html=True)

    if N_valid < 4:
        st.warning("样本量不足 (n<4)，无法进行系统误差检验")
    else:
        # 马利科夫判据（累进性系统误差）
        m = N_valid // 2
        D = np.sum(v_i_valid[:m]) - np.sum(v_i_valid[m:])

        st.markdown("### 马利科夫判据 (累进性系统误差)")

        if teaching_mode:
            st.markdown("""
            <div class="edu-box">
                <h4>马利科夫判据</h4>
                <p>将残差序列分为前后两半，计算差值 D。|D| 超过阈值时判定存在累进性系统误差。</p>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"D = \sum_{i=1}^{m} v_i - \sum_{i=m+1}^{n} v_i")
            st.latex(r"\text{其中 } m = \frac{n}{2}")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("判据值 D", f"{D:.4f}")
        with col2:
            sigma_D = sigma_bessel_valid * np.sqrt(N_valid / 2)
            if abs(D) > sigma_D:
                st.error(f"检验未通过：存在显著的累进性系统误差（|D| = {abs(D):.4f} > σᴅ = {sigma_D:.4f}）")
            else:
                st.success(f"检验通过：未发现累进性系统误差（|D| = {abs(D):.4f} ≤ σᴅ）")

        st.markdown("---")

        # 阿贝-赫梅特判据（周期性系统误差）
        sum_vv = np.sum(v_i_valid[:-1] * v_i_valid[1:])
        eta = abs(sum_vv) / (sigma_bessel_valid**2 * np.sqrt(N_valid - 1)) if sigma_bessel_valid > 0 else 0

        st.markdown("### 阿贝-赫梅特判据 (周期性系统误差)")

        if teaching_mode:
            st.markdown("""
            <div class="edu-box">
                <h4>阿贝-赫梅特判据</h4>
                <p>计算相邻残差的乘积和与根号下(n-1)乘s平方的比值 η。η > 1 时判定存在周期性系统误差。</p>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"\eta = \frac{\left|\sum_{i=1}^{n-1} v_i v_{i+1}\right|}{s^2\sqrt{n-1}}")

        col3, col4 = st.columns([1, 2])
        with col3:
            st.metric("判据值 η", f"{eta:.4f}")
        with col4:
            if eta > 1:
                st.error(f"检验未通过：存在显著的周期性系统误差（η = {eta:.4f} > 1）")
            else:
                st.success(f"检验通过：未发现周期性系统误差（η = {eta:.4f} ≤ 1）")

        if teaching_mode:
            st.markdown("""
            <div class="edu-box">
                <h4>为什么阿贝-赫梅特判据有时判别不出来？</h4>
                <p>周期性系统误差的检测灵敏度受以下因素影响：</p>
                <ul>
                    <li><b>样本量 n</b>：n 越小，η 的方差越大，检验功效越低，容易把真实的周期性误差误判为随机波动。</li>
                    <li><b>周期长度 T</b>：若测量周期 T 与样本量 n 不匹配（如 T 很大或 T 不是 n 的整因子），周期性特征不明显，η 难以捕捉。</li>
                    <li><b>随机误差掩盖</b>：当随机误差幅值 σ 较大时，周期性的起伏被噪声淹没，相邻残差乘积和有正有负，互相抵消，η 变小。</li>
                    <li><b>数据顺序</b>：如果周期性误差在测量前期占主导、后期被其他误差掩盖，判据也可能失效。</li>
                </ul>
                <p>建议：先观察残差时序图，若有明显的周期波动趋势再使用该判据。</p>
            </div>
            """, unsafe_allow_html=True)

        # 残差时序图
        st.markdown("---")
        st.markdown("#### 残差时序图")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=np.arange(1, N_valid + 1), y=v_i_valid,
            mode='lines+markers',
            name='残差',
            line=dict(color='#4A90D9', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(74,144,217,0.1)'
        ))
        fig3.add_hline(y=0, line_dash='dash', line_color='gray')
        fig3.update_layout(
            template='plotly_white',
            height=300,
            xaxis_title='测量序号 i',
            yaxis_title='残差 vᵢ'
        )
        st.plotly_chart(fig3, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 3: 粗大误差判别
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("## 粗大误差判别与剔除")

    # 确保 method 在 session_state 中有初始值
    if 'gross_method' not in st.session_state:
        st.session_state['gross_method'] = "莱特准则 (3σ)"

    # 每次生成新数据时，如果 raw_data 长度变了，重置为默认准则
    if 'prev_raw_data_len' not in st.session_state:
        st.session_state['prev_raw_data_len'] = 0
    current_len = len(st.session_state.get('raw_data', []))
    if current_len != st.session_state['prev_raw_data_len']:
        st.session_state['gross_method'] = "莱特准则 (3σ)"
        st.session_state['prev_raw_data_len'] = current_len
    # 数据长度没变但准则变了，保持当前准则
    elif st.session_state.get('raw_data') is not None:
        st.session_state['prev_raw_data_len'] = current_len

    if teaching_mode:
        st.markdown("""
        <div class="edu-box">
            <h4>莱特准则 (3σ 准则)</h4>
            <p>基于贝塞尔公式，当残差 |vᵢ| > 3s 时判定为粗大误差。适用于样本量 n > 10。</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"|v_i| > 3\sigma \quad \Rightarrow \quad \text{判定为粗大误差}")
        st.latex(r"\sigma = \sqrt{\frac{\sum v_i^2}{n-1}}")

        st.markdown("""
        <div class="edu-box">
            <h4>格拉布斯准则 (Grubbs' Test)</h4>
            <p>适用于小样本测量 (n ≤ 20) 的粗大误差判别，比 3σ 准则更严格可靠。</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"G_i = \frac{|v_i|}{s} > G(n, P) \quad \Rightarrow \quad \text{判定为粗大误差}")
        st.latex(r"G(n, P) = \frac{n-1}{\sqrt{n}} \sqrt{\frac{t_{P}^2}{n-2+t_{P}^2}}")
        st.caption("其中 tₚ 为 t 分布临界值，P 为置信概率，s 为贝塞尔公式标准差（s 即 σ 的估计）")

    # 判别方法选择
    method = st.selectbox(
        "选择判别准则",
        ["莱特准则 (3σ)", "格拉布斯准则 (Grubbs)"],
        index=0 if st.session_state['gross_method'] == "莱特准则 (3σ)" else 1,
        key="gross_method_select"
    )
    # 同步回 session_state
    st.session_state['gross_method'] = method

    # 使用格拉布斯准则时显示提示
    if teaching_mode and method == "格拉布斯准则 (Grubbs)":
        st.markdown("""
        <div class="edu-box">
            <h4>注意：原始随机数据的正常波动</h4>
            <p>格拉布斯准则比莱特准则更严格，即使在纯随机误差数据中，也可能因随机波动的正常概率分布而检出个别粗大误差点。这并非异常现象，属于正常检出概率范畴。</p>
        </div>
        """, unsafe_allow_html=True)

    # 执行迭代剔除
    if N < 3:
        st.warning("样本量不足 (n<3)，无法进行粗大误差检验")
    else:
        # 已确认剔除的历史值（跨注入保留）
        confirmed_removed_values = st.session_state.get('confirmed_removed_values', [])

        # 合并已确认的和新检出的，用于显示
        all_removed_values = list(confirmed_removed_values)
        newly_removed_values = []

        valid_mask, removal_history = iterative_gross_error_removal(x_valid, method)

        # 新检出的粗大误差值
        for r in removal_history:
            newly_removed_values.append(r['value'])

        # 可视化
        fig2 = go.Figure()

        # 用当前有效数据均值绘制基准线
        current_mean = np.mean(x_valid[valid_mask[:len(x_valid)]])

        # 在 x_i（原始完整数据）中的有效索引
        retained_raw_indices = []
        for idx in np.where(valid_mask)[0]:
            # x_valid[idx] 对应 x_i 中的哪个位置？
            # x_valid 是 cleaned_data，cleaned_data = x_i[removed_mask]
            # 需要找到 x_valid[idx] 在 x_i 中的原始索引
            val = x_valid[idx]
            matches = np.where(np.isclose(x_i, val))[0]
            if len(matches) > 0:
                retained_raw_indices.append(matches[0])

        # 被剔除的点（之前已确认的 + 新检出的）
        all_removed_raw_indices = []
        # 已确认剔除的点（从 x_i 中查找）
        for val in confirmed_removed_values:
            matches = np.where(np.isclose(x_i, val))[0]
            if len(matches) > 0:
                all_removed_raw_indices.append(matches[0])
        # 新检出的点（从 x_i 中查找）
        for val in newly_removed_values:
            matches = np.where(np.isclose(x_i, val))[0]
            if len(matches) > 0 and matches[0] not in all_removed_raw_indices:
                all_removed_raw_indices.append(matches[0])

        # 保留的点（在 x_i 中的原始索引）
        retained_raw_indices_set = set(range(len(x_i))) - set(all_removed_raw_indices)

        # 保留的点（用原始索引排序）
        retained_sorted = sorted(retained_raw_indices_set)
        fig2.add_trace(go.Scatter(
            x=np.array(retained_sorted) + 1,
            y=x_i[retained_sorted],
            mode='markers',
            name='有效数据',
            marker=dict(color='#4A90D9', size=12, symbol='circle-open')
        ))

        # 已剔除的点（用原始索引排序）
        removed_sorted = sorted(all_removed_raw_indices)
        if len(removed_sorted) > 0:
            fig2.add_trace(go.Scatter(
                x=np.array(removed_sorted) + 1,
                y=x_i[removed_sorted],
                mode='markers',
                name='已剔除',
                marker=dict(color='#EF4444', size=20, symbol='x', line=dict(width=3))
            ))

        fig2.add_hline(y=current_mean, line_dash='dash', line_color='gray',
                       annotation_text=f'均值 = {current_mean:.2f}')
        fig2.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title='测量序号 i',
            yaxis_title='测量值',
            hovermode='closest'
        )

        st.plotly_chart(fig2, use_container_width=True, key=f"gross_chart_{len(x_valid)}_{method}")

        # 判别结果统计（已确认的 + 新检出的）
        total_removed = len(confirmed_removed_values) + len(removal_history)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("检出粗大误差点数（累计）", total_removed)
        with col_b:
            st.metric("剩余有效数据", np.sum(valid_mask))

        # 计算过程详情
        if N >= 3:
            st.markdown("---")
            st.markdown("**计算过程**")

            if method == "莱特准则 (3σ)":
                n_check = len(x_valid)
                x_bar_check = np.mean(x_valid)
                residuals_check = x_valid - x_bar_check
                s_check = np.sqrt(np.sum(residuals_check**2) / (n_check - 1))
                threshold_check = 3 * s_check

                st.latex(r"\bar{x} = \frac{1}{n}\sum x_i = " + f"{x_bar_check:.4f}")
                st.latex(r"v_i = x_i - \bar{x}")
                st.latex(r"s = \sqrt{\frac{\sum v_i^2}{n-1}} = " + f"{s_check:.4f}")
                st.latex(r"3\sigma = 3s = " + f"{threshold_check:.4f}")
            else:
                n_check = len(x_valid)
                x_bar_check = np.mean(x_valid)
                residuals_check = x_valid - x_bar_check
                s_check = np.sqrt(np.sum(residuals_check**2) / (n_check - 1))
                t_crit = stats.t.ppf(0.975, n_check - 2)
                G_crit = (n_check - 1) / np.sqrt(n_check) * np.sqrt(t_crit**2 / (n_check - 2 + t_crit**2))
                threshold_check = G_crit * s_check

                st.latex(r"\bar{x} = \frac{1}{n}\sum x_i = " + f"{x_bar_check:.4f}")
                st.latex(r"v_i = x_i - \bar{x}")
                st.latex(r"s = \sqrt{\frac{\sum v_i^2}{n-1}} = " + f"{s_check:.4f}")
                st.latex(r"G(n,P) = \frac{n-1}{\sqrt{n}}\sqrt{\frac{t_P^2}{n-2+t_P^2}} = " + f"{G_crit:.4f}")
                st.latex(r"G(n,P) \cdot s = " + f"{threshold_check:.4f}")

            # 列出残差最大值与阈值比较
            max_residual = np.max(np.abs(residuals_check))
            max_idx = np.argmax(np.abs(residuals_check))
            st.latex(r"|v|_{\max} = " + f"{max_residual:.4f}" + r" \quad @\quad " + r"\text{索引 } i = " + f"{max_idx + 1}")

            if len(removal_history) > 0:
                st.markdown("**迭代剔除过程**")
                for i, r in enumerate(removal_history):
                    criterion_val = r['sigma'] if method == "莱特准则 (3σ)" else r['sigma']
                    st.markdown(f"- 第{i+1}次：|v|={abs(r['residual']):.4f} > {r['threshold']:.4f}，剔除索引{r['original_index']+1}，值={r['value']:.4f}")

        # 确认剔除按钮
        st.markdown("---")
        if st.button("确认剔除并重算", type="primary"):
            # 记录已确认剔除的值（用于跨注入显示）
            confirmed = st.session_state.get('confirmed_removed_values', [])
            for r in removal_history:
                if r['value'] not in confirmed:
                    confirmed.append(r['value'])
            st.session_state['confirmed_removed_values'] = confirmed

            # 更新 removed_mask
            new_mask = clean_mask.copy()
            for r in removal_history:
                new_mask[r['original_index']] = False

            st.session_state['removed_mask'] = new_mask
            st.session_state['cleaned_data'] = x_i[new_mask]
            st.session_state['removed_ever'] = True
            st.success(f"已剔除 {len(removal_history)} 个粗大误差点")
            st.rerun()
        elif len(removal_history) == 0 and N >= 3:
            st.success("检验通过：未检出粗大误差")


# -----------------------------------------------------------------------------
# Tab 4: 随机误差与最终结果
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("## 随机误差与不确定度分析")

    if teaching_mode:
        st.markdown("""
        <div class="edu-box">
            <h4>随机误差与置信区间</h4>
            <p>随机误差服从正态分布 N(μ, σ²)。根据置信概率 P，查 t 分布或正态分布得到扩展不确定度 U。</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"U = k \cdot \frac{\sigma}{\sqrt{n}} \quad \text{(k 为覆盖因子)}")
        st.latex(r"\text{例如：置信概率} P=0.95 \Rightarrow k=t_{0.975}(n-1) \text{（t分布）} \ 或 \ k=1.96 \text{（正态分布）}")

    if N_valid < 2:
        st.warning("有效数据不足，无法计算不确定度")
    else:
        # 基本统计量
        x_bar_out = np.mean(x_valid)
        s_out = sigma_bessel_valid

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("算术平均值 x̄", f"{x_bar_out:.4f}")
        with col2:
            st.metric("实验标准差 s", f"{s_out:.4f}")
        with col3:
            st.metric("有效样本量 n", N_valid)

        st.markdown("---")

        # 置信水平选择
        confidence = st.select_slider(
            "选择置信概率 P",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%"
        )

        # 计算扩展不确定度
        if N_valid <= 20:
            # t 分布
            t_p = stats.t.ppf((1 + confidence) / 2, N_valid - 1)
            dist_name = "t 分布"
        else:
            # 正态分布
            t_p = stats.norm.ppf((1 + confidence) / 2)
            dist_name = "正态分布"

        U = t_p * s_out / np.sqrt(N_valid)  # 扩展不确定度

        st.markdown("---")

        # 最终结果
        st.markdown("#### 最终测量结果")

        # 计算置信区间
        lower = x_bar_out - U
        upper = x_bar_out + U

        # 高亮显示结果
        render_result_box(
            f"A = {x_bar_out:.4f} ± {U:.4f}"
        )

        # 详细信息
        with st.expander("详细计算信息"):
            st.markdown(f"""
            | 参数 | 值 |
            |------|-----|
            | 算术平均值 x̄ | {x_bar_out:.4f} |
            | 实验标准差 s | {s_out:.4f} |
            | 样本量 n | {N_valid} |
            | 置信概率 P | {int(confidence*100)}% |
            | 分布类型 | {dist_name} |
            | 覆盖因子 k | {t_p:.4f} |
            | 扩展不确定度 U | {U:.4f} |
            | 置信区间 | [{lower:.4f}, {upper:.4f}] |
            """)


# =============================================================================
# 页脚
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    <p>测量误差与数据处理动态演示系统 | 面向电子测量技术教学</p>
</div>
""", unsafe_allow_html=True)