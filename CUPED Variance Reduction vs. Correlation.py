import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from tqdm import tqdm

def generate_synthetic_data(n_samples: int, 
                          effect_size: float,
                          noise_level: float) -> pd.DataFrame:
    """
    生成合成数据，通过调整噪声水平来影响相关性
    
    参数:
        n_samples: 样本数量
        effect_size: 处理效应大小
        noise_level: 噪声水平，值越大相关性越低
    """
    np.random.seed(None)
    
    # 生成基础数据
    pre_treatment = np.random.normal(0, 1, n_samples)
    
    # 基于pre_treatment生成post_treatment，添加可控噪声
    base_post = pre_treatment.copy()  # 完全相关的基础
    noise = np.random.normal(0, noise_level, n_samples)
    post_treatment = base_post + noise
    
    # 生成处理组标记
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # 添加处理效应
    post_treatment[treatment == 1] += effect_size
    
    # 标准化处理，确保方差结构合理
    post_treatment = (post_treatment - np.mean(post_treatment)) / np.std(post_treatment)
    pre_treatment = (pre_treatment - np.mean(pre_treatment)) / np.std(pre_treatment)
    
    return pd.DataFrame({
        'pre_treatment': pre_treatment,
        'post_treatment': post_treatment,
        'treatment': treatment
    })

def cuped(data: pd.DataFrame) -> tuple:
    """计算CUPED统计量"""
    # 计算线性系数
    X = sm.add_constant(data['pre_treatment'])
    model = sm.OLS(data['post_treatment'], X).fit()
    theta = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r_squared = model.rsquared  # 获取整体拟合的决定系数
    
    # 计算CUPED统计量
    pre_treatment_mean = data['pre_treatment'].mean()
    cuped_stats = data['post_treatment'] - theta * (data['pre_treatment'] - pre_treatment_mean)
    
    # 分别计算处理组和对照组的相关系数和决定系数
    control_mask = data['treatment'] == 0
    treatment_mask = data['treatment'] == 1
    
    control_pearson = np.corrcoef(
        data[control_mask]['pre_treatment'],
        data[control_mask]['post_treatment']
    )[0,1]
    
    treatment_pearson = np.corrcoef(
        data[treatment_mask]['pre_treatment'],
        data[treatment_mask]['post_treatment']
    )[0,1]
    
    # 计算各组方差
    control_var = data[control_mask]['post_treatment'].var()
    treatment_var = data[treatment_mask]['post_treatment'].var()
    total_var = control_var + treatment_var
    
    # 计算理论方差缩减
    theoretical_var_reduction = (
        (control_pearson**2 * control_var + 
         treatment_pearson**2 * treatment_var) / total_var
    )
    
    # 计算实际方差缩减
    control_cuped_var = cuped_stats[control_mask].var()
    treatment_cuped_var = cuped_stats[treatment_mask].var()
    actual_var_reduction = 1 - (control_cuped_var + treatment_cuped_var) / total_var
    
    # 准备返回值
    control_original = data[control_mask]['post_treatment']
    treatment_original = data[treatment_mask]['post_treatment']
    control_cuped = cuped_stats[control_mask]
    treatment_cuped = cuped_stats[treatment_mask]
    pearson_corr = np.corrcoef(data['pre_treatment'], data['post_treatment'])[0,1]
    
    # 返回所有需要的值
    return (control_original, control_cuped, 
            treatment_original, treatment_cuped,
            theta, intercept, pearson_corr,
            control_pearson, treatment_pearson,
            r_squared, theoretical_var_reduction,
            actual_var_reduction)

def validate_and_plot(data: pd.DataFrame, cuped_results: tuple, ax=None) -> dict:
    """验证结果并绘图"""
    (control_original, control_cuped, 
     treatment_original, treatment_cuped,
     theta, intercept, pearson_corr,
     control_pearson, treatment_pearson,
     r_squared, theoretical_var_reduction,
     actual_var_reduction) = cuped_results
    
    # 计算效应
    original_effect = treatment_original.mean() - control_original.mean()
    cuped_effect = treatment_cuped.mean() - control_cuped.mean()
    
    # 计算每个样本的效应量差异
    original_effects = treatment_original - control_original.mean()
    cuped_effects = treatment_cuped - control_cuped.mean()
    
    # 检验两种效应量是否有显著差异
    t_stat, p_value = stats.ttest_ind(original_effects, cuped_effects)
    
    # 绘图
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        ax1, ax2 = ax
    
    # 左图：散点图
    for group, color, label in zip([0, 1], ['blue', 'red'], ['Control', 'Treatment']):
        mask = data['treatment'] == group
        ax1.scatter(data.loc[mask, 'pre_treatment'], 
                   data.loc[mask, 'post_treatment'],
                   c=color, alpha=0.5, label=label)
    
    x_range = np.linspace(data['pre_treatment'].min(), data['pre_treatment'].max(), 100)
    y_pred = theta * x_range + intercept
    ax1.plot(x_range, y_pred, 'k--', 
             label=f'Regression (θ={theta:.3f}, ρ={pearson_corr:.3f})')
    
    ax1.set_xlabel('Pre-treatment')
    ax1.set_ylabel('Post-treatment')
    ax1.legend()
    ax1.set_title('Pre vs Post Treatment')
    
    # 右图：直方图
    ax2.hist(control_original, alpha=0.3, label='Control (Original)', bins=50, color='blue')
    ax2.hist(treatment_original, alpha=0.3, label='Treatment (Original)', bins=50, color='lightblue')
    ax2.hist(control_cuped, alpha=0.3, label='Control (CUPED)', bins=50, color='red')
    ax2.hist(treatment_cuped, alpha=0.3, label='Treatment (CUPED)', bins=50, color='pink')
    
    ax2.axvline(x=control_original.mean(), color='blue', linestyle='--', alpha=0.5)
    ax2.axvline(x=treatment_original.mean(), color='lightblue', linestyle='--', alpha=0.5)
    ax2.axvline(x=control_cuped.mean(), color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=treatment_cuped.mean(), color='pink', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.set_title(f'Value Distribution\n' + 
                  f'Control Pearson: {control_pearson:.3f}, Treatment Pearson: {treatment_pearson:.3f}\n' +
                  f'Overall R²: {r_squared:.3f}, min(r1,r2)²: {min(control_pearson, treatment_pearson)**2:.3f}\n' +
                  f'Actual Var Reduction: {actual_var_reduction:.3f}\n' +
                  f'Theoretical Var Reduction: {theoretical_var_reduction:.3f}\n' +
                  f'Original Effect: {original_effect:.3f}, CUPED Effect: {cuped_effect:.3f}')
    
    plt.tight_layout()
    
    return {
        'variance_reduction': actual_var_reduction,
        'theoretical_reduction': theoretical_var_reduction,
        'original_effect': original_effect,
        'cuped_effect': cuped_effect,
        'effect_difference_p': p_value,
        'control_pearson': control_pearson,
        'treatment_pearson': treatment_pearson,
        'r_squared': r_squared
    }

# 主程序修改
if __name__ == "__main__":
    N_EXPERIMENTS = 50
    NOISE_LEVELS = np.array([0.1, 1.0, 2.5, 5.0])
    
    results_data = {
        'noise_level': [],
        'actual_rho': [],
        'control_rho': [],
        'treatment_rho': [],
        'var_reduction': [],
        'theoretical_reduction': [],
        'p_value': [],
        'r_squared': []
    }
    
    last_experiments = {}
    
    with tqdm(total=len(NOISE_LEVELS) * N_EXPERIMENTS, desc="Running experiments") as pbar:
        for noise_level in NOISE_LEVELS:
            for i in range(N_EXPERIMENTS):
                data = generate_synthetic_data(
                    n_samples=5000,
                    effect_size=2.0,
                    noise_level=noise_level
                )
                
                cuped_results = cuped(data)
                validation = validate_and_plot(data, cuped_results)
                plt.close()
                
                results_data['noise_level'].append(noise_level)
                results_data['actual_rho'].append(cuped_results[6])  # pearson_corr
                results_data['control_rho'].append(cuped_results[7])  # control_pearson
                results_data['treatment_rho'].append(cuped_results[8])  # treatment_pearson
                results_data['var_reduction'].append(validation['variance_reduction'])
                results_data['theoretical_reduction'].append(validation['theoretical_reduction'])
                results_data['p_value'].append(validation['effect_difference_p'])
                results_data['r_squared'].append(validation['r_squared'])
                
                if i == N_EXPERIMENTS - 1:
                    last_experiments[str(noise_level)] = (data, cuped_results)
                
                pbar.update(1)
    
    # 转换为DataFrame并计算均值
    results_df = pd.DataFrame(results_data)
    noise_level_means = results_df.groupby('noise_level').mean()
    
    # 创建并显示汇总图
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：方差缩减 vs 噪声水平
    sns.boxplot(x='noise_level', 
                y='var_reduction', 
                data=results_df, 
                ax=ax1,
                width=0.5,
                color='blue',
                label='Actual')
    
    # 添加理论值折线
    ax1.plot(range(len(NOISE_LEVELS)), 
            noise_level_means['theoretical_reduction'],
            'r-', linewidth=2, marker='o', label='Theoretical')
    
    ax1.set_title('Variance Reduction vs Noise Level')
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Variance Reduction')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图：P值 vs 噪声水平
    sns.boxplot(x='noise_level', 
                y='p_value', 
                data=results_df, 
                ax=ax2,
                width=0.5)
    
    ax2.set_title('P-value vs Noise Level')
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('P-value')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存汇总图但不关闭
    plt.savefig('variance_reduction_summary.png', dpi=300, bbox_inches='tight')
    
    # 显示汇总图
    plt.show()
    
    # 然后再依次创建和显示详细分析图
    for noise_level in NOISE_LEVELS:
        if str(noise_level) in last_experiments:
            data, cuped_results = last_experiments[str(noise_level)]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 散点图
            for group, color, label in zip([0, 1], ['blue', 'red'], ['Control', 'Treatment']):
                mask = data['treatment'] == group
                ax1.scatter(data.loc[mask, 'pre_treatment'], 
                          data.loc[mask, 'post_treatment'],
                          c=color, alpha=0.5, label=label)
            
            theta = cuped_results[4]
            pearson_corr = cuped_results[6]
            control_pearson = cuped_results[7]
            treatment_pearson = cuped_results[8]
            r_squared = cuped_results[9]
            
            x_range = np.linspace(data['pre_treatment'].min(), data['pre_treatment'].max(), 100)
            y_pred = theta * x_range + cuped_results[5]
            ax1.plot(x_range, y_pred, 'k--', label='Regression Line')
            
            # 左图图例：基本信息
            basic_stats = (
                f'θ = {theta:.3f}\n'
                f'Overall ρ = {pearson_corr:.3f}\n'
                f'Control ρ = {control_pearson:.3f}\n'
                f'Treatment ρ = {treatment_pearson:.3f}\n'
                f'R² = {r_squared:.3f}'
            )
            
            ax1.text(0.02, 0.98, basic_stats,
                    transform=ax1.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top',
                    fontsize=10)
            
            ax1.set_xlabel('Pre-treatment')
            ax1.set_ylabel('Post-treatment')
            ax1.legend(loc='upper right')
            ax1.set_title(f'Pre vs Post Treatment (Noise Level = {noise_level})')
            ax1.grid(True, alpha=0.3)
            
            # 直方图
            control_original = cuped_results[0]
            treatment_original = cuped_results[2]
            control_cuped = cuped_results[1]
            treatment_cuped = cuped_results[3]
            
            ax2.hist(control_original, alpha=0.3, label='Control (Original)', bins=50, color='blue')
            ax2.hist(treatment_original, alpha=0.3, label='Treatment (Original)', bins=50, color='lightblue')
            ax2.hist(control_cuped, alpha=0.3, label='Control (CUPED)', bins=50, color='red')
            ax2.hist(treatment_cuped, alpha=0.3, label='Treatment (CUPED)', bins=50, color='pink')
            
            # 添加均值线
            ax2.axvline(x=control_original.mean(), color='blue', linestyle='--', alpha=0.5)
            ax2.axvline(x=treatment_original.mean(), color='lightblue', linestyle='--', alpha=0.5)
            ax2.axvline(x=control_cuped.mean(), color='red', linestyle='--', alpha=0.5)
            ax2.axvline(x=treatment_cuped.mean(), color='pink', linestyle='--', alpha=0.5)
            
            # 右图图例：关键结果
            theoretical_var_red = cuped_results[10]
            actual_var_red = cuped_results[11]
            original_effect = treatment_original.mean() - control_original.mean()
            cuped_effect = treatment_cuped.mean() - control_cuped.mean()
            p_value = validate_and_plot(data, cuped_results)["effect_difference_p"]
            
            results_stats = (
                f'Theoretical Variance Reduction: {theoretical_var_red:.3f}\n'
                f'Actual Variance Reduction: {actual_var_red:.3f}\n'
                f'Original Effect: {original_effect:.3f}\n'
                f'CUPED Effect: {cuped_effect:.3f}\n'
                f'P-value for CUPED vs. Original: {p_value:.3e}'
            )
            
            ax2.text(0.02, 0.98, results_stats,
                    transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top',
                    fontsize=10)
            
            ax2.set_title('Value Distribution')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Count')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'detailed_analysis_noise_{noise_level}.png', dpi=300, bbox_inches='tight')
            plt.show()  # 确保每个详细分析图都被显示
            plt.close()  # 关闭当前图形以释放内存
    
    # plt.show()