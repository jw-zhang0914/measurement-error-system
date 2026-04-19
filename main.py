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
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
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
                'index': max_residual_idx,
                'value': current_data[max_residual_idx],
                'residual': residuals[max_residual_idx],
                'threshold': threshold,
                'sigma': s if method == "莱特准则 (3σ)" else G_crit
            })
            current_data = np.delete(current_data, max_residual_idx)
        else:
            break

    # 构建原始数据的有效掩码
    valid_mask = np.ones(len(data), dtype=bool)
    for removed in removed_history:
        # 找到被剔除元素在原始数据中的位置
        original_idx = removed['index']
        valid_mask[original_idx] = False

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
        N = st.slider("样本数量 N", 5, 50, 20)

        # 随机误差
        st.markdown("**随机误差 (σ)**")
        sigma = st.slider("σ 标准差", 0.1, 10.0, 1.0, 0.1)

        # 系统误差
        st.markdown("**系统误差**")
        C = st.slider("恒值系统误差 C", -10.0, 10.0, 0.0, 0.1)
        K = st.slider("累进系统误差 K (线性漂移)", -1.0, 1.0, 0.0, 0.01)
        A_sys = st.slider("周期系统误差幅值 A", 0.0, 10.0, 0.0, 0.1)
        T = st.slider("周期系统误差周期 T", 1.0, 20.0, 10.0, 0.5)

        # 粗大误差
        st.markdown("**粗大误差注入**")
        gross_count = st.slider("突变点个数", 0, 5, 0)
        gross_magnitude = st.slider("偏离倍数 (σ)", 3.0, 10.0, 5.0, 0.5)

        generate_btn = st.button("生成模拟数据", type="primary", use_container_width=True)

        if generate_btn:
            # 生成无误差基准
            baseline = np.full(N, A0)

            # 添加随机误差（正态分布）
            rng = np.random.default_rng()
            random_err = rng.normal(0, sigma, N)

            # 添加系统误差
            i = np.arange(1, N + 1)
            systematic_err = C + K * i + A_sys * np.sin(2 * np.pi * i / T)

            # 合成含误差数据
            raw_data = baseline + random_err + systematic_err

            # 注入粗大误差
            if gross_count > 0:
                gross_indices = rng.choice(N, size=min(gross_count, N), replace=False)
                raw_data[gross_indices] += rng.choice([-1, 1], gross_count) * gross_magnitude * sigma

            # 存入 session_state
            st.session_state['raw_data'] = raw_data
            st.session_state['baseline'] = baseline
            st.session_state['cleaned_data'] = raw_data.copy()
            st.session_state['removed_mask'] = np.ones(N, dtype=bool)
            st.session_state['params'] = {
                'A0': A0, 'N': N, 'sigma': sigma,
                'C': C, 'K': K, 'A_sys': A_sys, 'T': T
            }
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

tab1, tab2, tab3, tab4 = st.tabs(["数据与残差", "粗大误差剔除", "系统误差检验", "随机误差分析"])


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
# Tab 2: 粗大误差判别
# -----------------------------------------------------------------------------
with tab2:
    st.markdown("## 粗大误差判别与剔除")

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
        ["莱特准则 (3σ)", "格拉布斯准则 (Grubbs)"]
    )

    # 执行迭代剔除
    if N < 3:
        st.warning("样本量不足 (n<3)，无法进行粗大误差检验")
    else:
        valid_mask, removal_history = iterative_gross_error_removal(x_valid, method)

        # 可视化
        fig2 = go.Figure()

        # 用当前有效数据均值绘制基准线
        current_mean = np.mean(x_valid[valid_mask[:len(x_valid)]])

        # 标记被剔除的点（红色X）
        removed_indices = np.where(~valid_mask)[0]
        retained_indices = np.where(valid_mask)[0]

        # 保留的点
        fig2.add_trace(go.Scatter(
            x=retained_indices + 1, y=x_valid[retained_indices],
            mode='markers',
            name='有效数据',
            marker=dict(color='#4A90D9', size=12, symbol='circle-open')
        ))

        # 被剔除的点
        if len(removed_indices) > 0:
            fig2.add_trace(go.Scatter(
                x=removed_indices + 1, y=x_valid[removed_indices],
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

        st.plotly_chart(fig2, use_container_width=True)

        # 判别结果统计
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("检出粗大误差点数", len(removal_history))
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
                    st.markdown(f"- 第{i+1}次：|v|={abs(r['residual']):.4f} > {r['threshold']:.4f}，剔除索引{r['index']+1}，值={r['value']:.4f}")

        # 确认剔除按钮
        st.markdown("---")
        if st.button("确认剔除并重算", type="primary"):
            # 构建新的 session_state
            new_mask = clean_mask.copy()
            for r in removal_history:
                # 找到对应原始数据中的位置
                new_mask[np.where((x_i == r['value']))[0][0] if len(np.where(x_i == r['value'])[0]) > 0 else 0] = False

            st.session_state['removed_mask'] = new_mask
            st.session_state['cleaned_data'] = x_i[new_mask]
            st.session_state['removed_ever'] = True
            st.success(f"已剔除 {len(removal_history)} 个粗大误差点")
            st.rerun()
        elif len(removal_history) == 0 and N >= 3:
            st.success("检验通过：未检出粗大误差")


# -----------------------------------------------------------------------------
# Tab 3: 系统误差判别
# -----------------------------------------------------------------------------
with tab3:
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
        eta = abs(sum_vv) / ((N_valid - 1) * sigma_bessel_valid**2) if sigma_bessel_valid > 0 else 0

        st.markdown("### 阿贝-赫梅特判据 (周期性系统误差)")

        if teaching_mode:
            st.markdown("""
            <div class="edu-box">
                <h4>阿贝-赫梅特判据</h4>
                <p>计算相邻残差的乘积和与标准差平方的比值 η。η > 1 时判定存在周期性系统误差。</p>
            </div>
            """, unsafe_allow_html=True)
            st.latex(r"\eta = \frac{\left|\sum_{i=1}^{n-1} v_i v_{i+1}\right|}{(n-1)s^2}")

        col3, col4 = st.columns([1, 2])
        with col3:
            st.metric("判据值 η", f"{eta:.4f}")
        with col4:
            if eta > 1:
                st.error(f"检验未通过：存在显著的周期性系统误差（η = {eta:.4f} > 1）")
            else:
                st.success(f"检验通过：未发现周期性系统误差（η = {eta:.4f} ≤ 1）")

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