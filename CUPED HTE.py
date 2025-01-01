import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import seaborn as sns
from typing import Dict, Tuple, Optional, Union

def generate_synthetic_data(n_samples: int = 5000, 
                          noise_level: float = 1.0,
                          random_state: Optional[int] = None) -> pd.DataFrame:
    """
    生成具有异质效应的合成数据
    
    参数:
        n_samples: 样本量
        noise_level: 噪声水平
        random_state: 随机种子
    
    返回:
        包含pre_treatment、post_treatment、treatment、segment的DataFrame
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 生成基础数据
    pre_treatment = np.random.normal(0, 1, n_samples)
    treatment = np.random.binomial(1, 0.5, n_samples)
    segment = np.random.randint(0, 3, n_samples)
    
    # 定义异质效应参数
    slopes = {0: 0.5, 1: 1, 2: 1.5}  # 不同segment的斜率
    effects = {0: 1.0, 1: 2.0, 2: 3.0}  # 不同segment的处理效应
    
    # 生成post_treatment
    noise = np.random.normal(0, noise_level, n_samples)
    base_effect = np.array([slopes[seg] * pre for seg, pre in zip(segment, pre_treatment)])
    treatment_effect = np.array([effects[seg] if treat == 1 else 0 
                               for seg, treat in zip(segment, treatment)])
    
    post_treatment = base_effect + treatment_effect + noise
    
    # 创建数据框
    data = pd.DataFrame({
        'pre_treatment': pre_treatment,
        'post_treatment': post_treatment,
        'treatment': treatment,
        'segment': segment
    })
    
    # 打印数据生成参数
    print("\nData Generation Parameters:")
    print(f"Sample size: {n_samples}")
    print(f"Slopes by segment: {slopes}")
    print(f"Treatment effects by segment: {effects}")
    print(f"Noise level: {noise_level}")
    
    return data

def run_cuped(data: pd.DataFrame, 
              multivariate: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """运行CUPED分析"""
    # 构建特征矩阵
    X = pd.DataFrame()
    if multivariate:
        # 多元线性模型：为每个segment创建独立的pre_treatment系数
        for seg in data['segment'].unique():
            mask = (data['segment'] == seg)
            X[f'pre_treatment_seg{seg}'] = data['pre_treatment'] * mask
        
        # 添加segment哑变量（除了基准组）
        segments = sorted(data['segment'].unique())[1:]
        for seg in segments:
            X[f'segment_{seg}'] = (data['segment'] == seg).astype(int)
    else:
        # 一元线性模型：只使用pre_treatment
        X['pre_treatment'] = data['pre_treatment']
    
    # 拟合模型
    X = sm.add_constant(X)
    model = sm.OLS(data['post_treatment'], X).fit()
    
    # 初始化结果
    results = {
        'model': model,
        'model_params': model.params.to_dict(),
        'model_r2': model.rsquared,
        'overall': {},
        'segments': {}
    }
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'post_treatment_orig': data['post_treatment'],
        'post_treatment_cuped': data['post_treatment'].copy()
    })
    
    # CUPED处理
    adjusted_values = data['post_treatment'].copy()
    
    if multivariate:
        # 计算CUPED调整值
        for seg in data['segment'].unique():
            mask = (data['segment'] == seg)
            theta = model.params[f'pre_treatment_seg{seg}']
            x_mean = data.loc[mask, 'pre_treatment'].mean()
            adjusted_values[mask] -= theta * (data.loc[mask, 'pre_treatment'] - x_mean)
        
        # 计算总体统计量
        total_weighted_r2 = 0
        total_weighted_var = 0
        total_var_orig = 0
        total_var_cuped = 0
        total_n = 0
        
        # 计算分组统计量
        for seg in data['segment'].unique():
            mask = (data['segment'] == seg)
            seg_weighted_r2 = 0
            seg_weighted_var = 0
            seg_var_orig = 0
            seg_var_cuped = 0
            seg_n = 0
            
            # 分别计算处理组和对照组
            for treat in [0, 1]:
                treat_mask = mask & (data['treatment'] == treat)
                pre = data.loc[treat_mask, 'pre_treatment']
                post = data.loc[treat_mask, 'post_treatment']
                cuped = adjusted_values[treat_mask]
                
                n_treat = len(post)
                var_orig_treat = np.var(post)
                var_cuped_treat = np.var(cuped)
                r = np.corrcoef(pre, post)[0,1] if len(pre) > 1 else 0
                
                # 累加分组统计量
                seg_weighted_r2 += n_treat * r**2 * var_orig_treat
                seg_weighted_var += n_treat * var_orig_treat
                seg_var_orig += var_orig_treat * n_treat
                seg_var_cuped += var_cuped_treat * n_treat
                seg_n += n_treat
                
                # 累加总体统计量
                total_weighted_r2 += n_treat * r**2 * var_orig_treat
                total_weighted_var += n_treat * var_orig_treat
                total_var_orig += var_orig_treat * n_treat
                total_var_cuped += var_cuped_treat * n_treat
                total_n += n_treat
            
            # 计算分组方差缩减
            results['segments'][seg] = {
                'var_orig': seg_var_orig / seg_n,
                'var_cuped': seg_var_cuped / seg_n,
                'theoretical_reduction': seg_weighted_r2 / seg_weighted_var,
                'actual_reduction': 1 - (seg_var_cuped / seg_var_orig),
                'effect_orig': np.mean(data.loc[mask & (data['treatment'] == 1), 'post_treatment']) - 
                             np.mean(data.loc[mask & (data['treatment'] == 0), 'post_treatment']),
                'effect_cuped': np.mean(adjusted_values[mask & (data['treatment'] == 1)]) - 
                               np.mean(adjusted_values[mask & (data['treatment'] == 0)]),
                'effect_diff_p_value': 2 * (1 - scipy_stats.t.cdf(abs(
                    np.mean(adjusted_values[mask & (data['treatment'] == 1)]) - 
                    np.mean(adjusted_values[mask & (data['treatment'] == 0)]) - 
                    (np.mean(data.loc[mask & (data['treatment'] == 1), 'post_treatment']) - 
                     np.mean(data.loc[mask & (data['treatment'] == 0), 'post_treatment']))
                ), len(data[mask]) - 2))
            }
        
        # 计算总体方差缩减
        results['overall'] = {
            'var_orig': total_var_orig / total_n,
            'var_cuped': total_var_cuped / total_n,
            'theoretical_reduction': total_weighted_r2 / total_weighted_var,
            'actual_reduction': 1 - (total_var_cuped / total_var_orig),
            'effect_orig': np.mean(data[data['treatment'] == 1]['post_treatment']) - 
                         np.mean(data[data['treatment'] == 0]['post_treatment']),
            'effect_cuped': np.mean(adjusted_values[data['treatment'] == 1]) - 
                           np.mean(adjusted_values[data['treatment'] == 0]),
            'effect_diff_p_value': 2 * (1 - scipy_stats.t.cdf(abs(
                np.mean(adjusted_values[data['treatment'] == 1]) - 
                np.mean(adjusted_values[data['treatment'] == 0]) - 
                (np.mean(data[data['treatment'] == 1]['post_treatment']) - 
                 np.mean(data[data['treatment'] == 0]['post_treatment']))
            ), len(data) - 2))
        }
    
    else:
        # 一元CUPED处理
        theta = model.params['pre_treatment']
        x_mean = data['pre_treatment'].mean()
        
        # 计算CUPED调整值
        adjusted_values = data['post_treatment'] - theta * (data['pre_treatment'] - x_mean)
        
        # 计算整体统计量
        r_global = np.corrcoef(data['pre_treatment'], data['post_treatment'])[0,1]
        var_orig = np.var(data['post_treatment'])
        var_cuped = np.var(adjusted_values)
        
        # 计算效应量差异的p值
        ctrl_orig = data[data['treatment'] == 0]['post_treatment']
        ctrl_cuped = adjusted_values[data['treatment'] == 0]
        treat_orig = data[data['treatment'] == 1]['post_treatment']
        treat_cuped = adjusted_values[data['treatment'] == 1]
        
        se_orig = np.sqrt(np.var(treat_orig)/len(treat_orig) + 
                         np.var(ctrl_orig)/len(ctrl_orig))
        se_cuped = np.sqrt(np.var(treat_cuped)/len(treat_cuped) + 
                          np.var(ctrl_cuped)/len(ctrl_cuped))
        effect_orig = np.mean(treat_orig) - np.mean(ctrl_orig)
        effect_cuped = np.mean(treat_cuped) - np.mean(ctrl_cuped)
        effect_diff = effect_cuped - effect_orig
        se_diff = np.sqrt(se_cuped**2 + se_orig**2)
        t_stat = effect_diff / se_diff
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), len(data) - 2))
        
        results['overall'] = {
            'var_orig': var_orig,
            'var_cuped': var_cuped,
            'theoretical_reduction': r_global ** 2,
            'actual_reduction': 1 - (var_cuped / var_orig),
            'effect_orig': effect_orig,
            'effect_cuped': effect_cuped,
            'effect_diff_p_value': p_value
        }
        
        # 计算分组统计量
        for seg in data['segment'].unique():
            mask = (data['segment'] == seg)
            group_data = data[mask]
            
            # 计算组内统计量
            var_x = np.var(group_data['pre_treatment'])
            var_y = np.var(group_data['post_treatment'])
            cov_xy = np.cov(group_data['pre_treatment'], 
                          group_data['post_treatment'])[0,1]
            
            # 理论方差缩减
            theoretical_reduction = 1 - (var_y + theta**2 * var_x - 
                                      2 * theta * cov_xy) / var_y
            
            # 实际方差缩减
            var_cuped = np.var(adjusted_values[mask])
            
            # 计算分组效应量差异的p值
            ctrl_orig = data.loc[mask & (data['treatment'] == 0), 'post_treatment']
            ctrl_cuped = adjusted_values[mask & (data['treatment'] == 0)]
            treat_orig = data.loc[mask & (data['treatment'] == 1), 'post_treatment']
            treat_cuped = adjusted_values[mask & (data['treatment'] == 1)]
            
            se_orig = np.sqrt(np.var(treat_orig)/len(treat_orig) + 
                            np.var(ctrl_orig)/len(ctrl_orig))
            se_cuped = np.sqrt(np.var(treat_cuped)/len(treat_cuped) + 
                             np.var(ctrl_cuped)/len(ctrl_cuped))
            effect_orig = np.mean(treat_orig) - np.mean(ctrl_orig)
            effect_cuped = np.mean(treat_cuped) - np.mean(ctrl_cuped)
            effect_diff = effect_cuped - effect_orig
            se_diff = np.sqrt(se_cuped**2 + se_orig**2)
            t_stat = effect_diff / se_diff
            p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), len(data[mask]) - 2))
            
            results['segments'][seg] = {
                'var_orig': var_y,
                'var_cuped': var_cuped,
                'theoretical_reduction': theoretical_reduction,
                'actual_reduction': 1 - (var_cuped / var_y),
                'effect_orig': effect_orig,
                'effect_cuped': effect_cuped,
                'effect_diff_p_value': p_value
            }
    
    # 更新CUPED处理后的值
    result_df['post_treatment_cuped'] = adjusted_values
    
    return result_df, results

def plot_cuped_results(data: pd.DataFrame, 
                      result_df: pd.DataFrame,
                      results: Dict,
                      model: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    """
    绘制CUPED结果的可视化图表
    
    参数:
        data: 原始数据
        result_df: CUPED处理后的数据框
        results: CUPED统计结果
        model: 拟合的模型
    """
    # 创建2x2的子图布局
    fig = plt.figure(figsize=(20, 18))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.3)
    
    # 散点图（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#1f77b4', '#2ca02c', '#d62728']  # 深色
    light_colors = ['#a6cee3', '#b2df8a', '#fb9a99']  # 浅色
    
    for seg in data['segment'].unique():
        seg_data = data[data['segment'] == seg]
        
        # 使用深浅色区分实验组和对照组
        for treat, alpha in zip([0, 1], [0.6, 0.6]):
            mask = seg_data['treatment'] == treat
            color = light_colors[int(seg)] if treat == 0 else colors[int(seg)]
            label = f'{"Treatment" if treat else "Control"} (Segment {seg})'
            ax1.scatter(seg_data.loc[mask, 'pre_treatment'], 
                       seg_data.loc[mask, 'post_treatment'],
                       c=color, 
                       alpha=alpha, 
                       label=label)
        
        # 拟合线
        x_range = np.linspace(data['pre_treatment'].min(), 
                            data['pre_treatment'].max(), 100)
        if 'pre_treatment' in model.params:
            # 一元线性模型
            theta = model.params['pre_treatment']
            intercept = model.params['const']
        else:
            # 多元线性模型
            theta = model.params[f'pre_treatment_seg{int(seg)}']
            intercept = model.params['const']
            if seg > 0:
                intercept += model.params[f'segment_{int(seg)}']
        
        y_pred = intercept + theta * x_range
        ax1.plot(x_range, y_pred, c=colors[int(seg)], linestyle='--',
                label=f'Fit (Segment {seg})')
    
    ax1.set_xlabel('Pre-treatment')
    ax1.set_ylabel('Post-treatment')
    ax1.set_title('Pre vs Post Treatment')
    ax1.grid(True, alpha=0.3)
    # 修改图例位置和样式
    ax1.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
              fontsize=8, framealpha=0.8)
    
    # 分布图
    plot_positions = [(0,1), (1,0), (1,1)]
    
    for seg in data['segment'].unique():
        ax = fig.add_subplot(gs[plot_positions[int(seg)]])
        mask = (data['segment'] == seg)
        
        # 获取数据
        ctrl_orig = result_df.loc[mask & (data['treatment'] == 0), 'post_treatment_orig']
        ctrl_cuped = result_df.loc[mask & (data['treatment'] == 0), 'post_treatment_cuped']
        treat_orig = result_df.loc[mask & (data['treatment'] == 1), 'post_treatment_orig']
        treat_cuped = result_df.loc[mask & (data['treatment'] == 1), 'post_treatment_cuped']
        
        # 调整直方图和密度曲线
        bins = np.linspace(
            min(result_df.loc[mask, 'post_treatment_orig']),
            max(result_df.loc[mask, 'post_treatment_orig']),
            60
        )
        
        # 控制组
        ax.hist(ctrl_orig, bins=bins, color='gray', alpha=0.3, 
               label='Control (Orig)', density=True)
        ax.hist(ctrl_cuped, bins=bins, color='lightgray', alpha=0.3,
               label='Control (CUPED)', density=True)
        
        # 实验组
        ax.hist(treat_orig, bins=bins, color=colors[int(seg)], alpha=0.3,
               label='Treatment (Orig)', density=True)
        ax.hist(treat_cuped, bins=bins, color=light_colors[int(seg)], alpha=0.3,
               label='Treatment (CUPED)', density=True)
        
        # 密度曲线
        for data_array, color, style in [
            (ctrl_orig, 'gray', '-'),
            (ctrl_cuped, 'gray', '--'),
            (treat_orig, colors[int(seg)], '-'),
            (treat_cuped, colors[int(seg)], '--')
        ]:
            kde = scipy_stats.gaussian_kde(data_array)
            x_range = np.linspace(min(bins), max(bins), 200)
            density = kde(x_range)
            density = density * (ax.get_ylim()[1] / density.max())
            ax.plot(x_range, density, color=color, linestyle=style, alpha=0.8)
        
        ax.set_title(f'Segment {seg} Distribution')
        ax.legend(fontsize=8)
        
        # 统计信息
        seg_stats = results['segments'][int(seg)]
        stats_text = (
            f"Theoretical Reduction: {seg_stats['theoretical_reduction']:.3f}\n"
            f"Actual Reduction: {seg_stats['actual_reduction']:.3f}\n"
            f"Original Effect: {seg_stats['effect_orig']:.3f}\n"
            f"CUPED Effect: {seg_stats['effect_cuped']:.3f}\n"
            f"Effect p-value: {seg_stats['effect_diff_p_value']:.3e}"
        )
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成测试数据
    data = generate_synthetic_data(n_samples=5000, noise_level=0.1, random_state=42)
    
    # 运行一元和多元CUPED
    print("\n=== 运行一元CUPED ===")
    uni_df, uni_results = run_cuped(data, multivariate=False)
    print("\n模型参数:")
    for param, value in uni_results['model_params'].items():
        print(f"{param}: {value:.3f}")
    print(f"R-squared: {uni_results['model_r2']:.3f}")
    
    print("\n=== 运行多元CUPED ===")
    multi_df, multi_results = run_cuped(data, multivariate=True)
    print("\n模型参数:")
    for param, value in multi_results['model_params'].items():
        print(f"{param}: {value:.3f}")
    print(f"R-squared: {multi_results['model_r2']:.3f}")
    
    # 比较结果
    print("\n=== CUPED方法比较 ===")
    methods = {'一元CUPED': (uni_df, uni_results), 
              '多元CUPED': (multi_df, multi_results)}
    
    for name, (df, results) in methods.items():
        print(f"\n{name}整体结果:")
        print(f"原始方差: {results['overall']['var_orig']:.3f}")
        print(f"CUPED方差: {results['overall']['var_cuped']:.3f}")
        print(f"理论方差缩减: {results['overall']['theoretical_reduction']:.3f}")
        print(f"实际方差缩减: {results['overall']['actual_reduction']:.3f}")
        print(f"原始效应量: {results['overall']['effect_orig']:.3f}")
        print(f"CUPED效应量: {results['overall']['effect_cuped']:.3f}")
        print(f"效应量差异p值: {results['overall']['effect_diff_p_value']:.3e}")
        
        for seg in sorted(results['segments'].keys()):
            print(f"\nSegment {seg}结果:")
            seg_stats = results['segments'][seg]
            print(f"原始方差: {seg_stats['var_orig']:.3f}")
            print(f"CUPED方差: {seg_stats['var_cuped']:.3f}")
            print(f"理论方差缩减: {seg_stats['theoretical_reduction']:.3f}")
            print(f"实际方差缩减: {seg_stats['actual_reduction']:.3f}")
            print(f"原始效应量: {seg_stats['effect_orig']:.3f}")
            print(f"CUPED效应量: {seg_stats['effect_cuped']:.3f}")
            print(f"效应量差异p值: {seg_stats['effect_diff_p_value']:.3e}")
    
    # 绘制结果
    print("\n=== 绘制一元CUPED结果 ===")
    plot_cuped_results(data, uni_df, uni_results, uni_results['model'])
    
    print("\n=== 绘制多元CUPED结果 ===")
    plot_cuped_results(data, multi_df, multi_results, multi_results['model'])


