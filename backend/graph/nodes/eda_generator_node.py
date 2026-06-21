import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from backend.config import settings
from backend.graph.state import PreprocessingGraphState
from backend.models.schemas import CorrelationMatrix, EDAResults, EDAStatistics, OutlierAnalysis
from backend.utils.exceptions import EDAGenerationException
logger = logging.getLogger(__name__)

class EDAGenerator:

    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_plots: List[str] = []
        try:
            plt.style.use(settings.PLOT_STYLE)
        except OSError:
            plt.style.use('seaborn-v0_8')
        except Exception:
            pass
        sns.set_theme(style='darkgrid', palette='viridis', font_scale=1.1)

    def generate_descriptive_statistics(self) -> List[EDAStatistics]:
        logger.info('Generating descriptive statistics...')
        stats_list: List[EDAStatistics] = []
        for col in self.df.columns:
            col_data = self.df[col]
            stat = EDAStatistics(column_name=col, count=int(col_data.count()), missing=int(col_data.isna().sum()), unique=int(col_data.nunique()))
            if pd.api.types.is_numeric_dtype(col_data):
                non_null = col_data.dropna()
                if len(non_null) > 0:
                    stat.mean = round(float(non_null.mean()), 4)
                    stat.std = round(float(non_null.std()), 4)
                    stat.min = round(float(non_null.min()), 4)
                    stat.percentile_25 = round(float(non_null.quantile(0.25)), 4)
                    stat.median = round(float(non_null.median()), 4)
                    stat.percentile_75 = round(float(non_null.quantile(0.75)), 4)
                    stat.max = round(float(non_null.max()), 4)
            else:
                mode_vals = col_data.mode()
                if len(mode_vals) > 0:
                    stat.mode = str(mode_vals.iloc[0])
                    stat.mode_frequency = int((col_data == mode_vals.iloc[0]).sum())
            stats_list.append(stat)
        logger.info(f'  Generated statistics for {len(stats_list)} columns')
        return stats_list

    def generate_correlation_matrix(self) -> Optional[CorrelationMatrix]:
        logger.info('Generating correlation matrix...')
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            logger.info('  Fewer than 2 numeric columns, skipping correlation matrix')
            return None
        corr_matrix = numeric_df.corr()
        corr_matrix = corr_matrix.fillna(0)
        result = CorrelationMatrix(columns=list(corr_matrix.columns), matrix=[[round(float(val), 4) for val in row] for row in corr_matrix.values])
        logger.info(f'  Correlation matrix: {len(result.columns)}x{len(result.columns)}')
        return result

    def analyze_missing_values(self) -> Dict[str, int]:
        logger.info('Analyzing missing values...')
        missing_counts = self.df.isna().sum()
        missing_dict = {col: int(count) for (col, count) in missing_counts.items() if count > 0}
        total_missing = sum(missing_dict.values())
        logger.info(f'  {len(missing_dict)} columns with missing values, {total_missing} total missing')
        return missing_dict

    def analyze_outliers(self) -> List[OutlierAnalysis]:
        logger.info('Analyzing outliers...')
        outlier_results: List[OutlierAnalysis] = []
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                continue
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)
            outlier_count = int(outlier_mask.sum())
            outlier_pct = round(outlier_count / len(non_null) * 100, 2)
            outlier_results.append(OutlierAnalysis(column_name=col, outlier_count=outlier_count, outlier_percentage=outlier_pct, lower_bound=round(float(lower_bound), 4), upper_bound=round(float(upper_bound), 4)))
        logger.info(f'  Analyzed outliers for {len(outlier_results)} numeric columns')
        return outlier_results

    def _save_plot(self, fig: plt.Figure, filename: str) -> str:
        filepath = self.output_dir / filename
        fig.savefig(str(filepath), dpi=settings.PLOT_DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        abs_path = str(filepath.resolve())
        self.generated_plots.append(abs_path)
        logger.debug(f'  Saved plot: {abs_path}')
        return abs_path

    def generate_correlation_heatmap(self) -> Optional[str]:
        if not settings.GENERATE_CORRELATION_HEATMAP:
            return None
        logger.info('Generating correlation heatmap...')
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            logger.info('  Not enough numeric columns for heatmap')
            return None
        corr = numeric_df.corr()
        n_cols = len(corr.columns)
        fig_size = max(8, min(20, n_cols * 0.8))
        (fig, ax) = plt.subplots(figsize=(fig_size, fig_size * 0.8))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=n_cols <= 15, fmt='.2f' if n_cols <= 15 else '', cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlation'}, ax=ax)
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        path = self._save_plot(fig, 'correlation_heatmap.png')
        logger.info(f'  Heatmap saved: {path}')
        return path

    def generate_distribution_plots(self) -> List[str]:
        if not settings.GENERATE_DISTRIBUTION_PLOTS:
            return []
        logger.info('Generating distribution plots...')
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            logger.info('  No numeric columns for distribution plots')
            return []
        saved_paths: List[str] = []
        n_cols_per_row = 3
        n_rows = max(1, (len(numeric_cols) + n_cols_per_row - 1) // n_cols_per_row)
        fig_height = n_rows * 4
        (fig, axes) = plt.subplots(n_rows, n_cols_per_row, figsize=(15, fig_height))
        if n_rows == 1 and n_cols_per_row == 1:
            axes_flat = [axes]
        elif n_rows == 1 or n_cols_per_row == 1:
            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes_flat = axes.flatten()
        for (idx, col) in enumerate(numeric_cols):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                try:
                    sns.histplot(non_null, kde=True, ax=ax, color='#4C72B0', edgecolor='white', alpha=0.7)
                except Exception:
                    ax.hist(non_null, bins=30, color='#4C72B0', edgecolor='white', alpha=0.7)
            ax.set_title(col, fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Count')
        for idx in range(len(numeric_cols), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = self._save_plot(fig, 'feature_distributions.png')
        saved_paths.append(path)
        logger.info(f'  Distribution plots saved: {path}')
        return saved_paths

    def generate_missing_value_plot(self) -> Optional[str]:
        if not settings.GENERATE_MISSING_VALUE_PLOTS:
            return None
        logger.info('Generating missing value plot...')
        missing = self.df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) == 0:
            logger.info('  No missing values to plot')
            return None
        (fig, ax) = plt.subplots(figsize=(12, max(4, len(missing) * 0.4)))
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(missing)))
        bars = ax.barh(range(len(missing)), missing.values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index, fontsize=10)
        ax.set_xlabel('Missing Value Count', fontsize=12)
        ax.set_title('Missing Values by Column', fontsize=16, fontweight='bold', pad=15)
        for (bar_item, value) in zip(bars, missing.values):
            pct = value / len(self.df) * 100
            ax.text(bar_item.get_width() + max(missing.values) * 0.01, bar_item.get_y() + bar_item.get_height() / 2, f'{value} ({pct:.1f}%)', va='center', fontsize=9)
        ax.invert_yaxis()
        fig.tight_layout()
        path = self._save_plot(fig, 'missing_values.png')
        logger.info(f'  Missing value plot saved: {path}')
        return path

    def generate_outlier_plots(self, outlier_analysis: List[OutlierAnalysis]) -> Optional[str]:
        if not settings.GENERATE_OUTLIER_PLOTS:
            return None
        logger.info('Generating outlier plots...')
        cols_with_outliers = [oa.column_name for oa in outlier_analysis if oa.outlier_count > 0 and oa.column_name in self.df.columns]
        if not cols_with_outliers:
            logger.info('  No columns with outliers to plot')
            return None
        cols_with_outliers = cols_with_outliers[:12]
        n_cols_per_row = 3
        n_rows = max(1, (len(cols_with_outliers) + n_cols_per_row - 1) // n_cols_per_row)
        (fig, axes) = plt.subplots(n_rows, n_cols_per_row, figsize=(15, n_rows * 4))
        if n_rows == 1 and n_cols_per_row == 1:
            axes_flat = [axes]
        elif n_rows == 1 or n_cols_per_row == 1:
            axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes_flat = axes.flatten()
        for (idx, col) in enumerate(cols_with_outliers):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            non_null = self.df[col].dropna()
            if len(non_null) > 0:
                bp = ax.boxplot(non_null, patch_artist=True, boxprops=dict(facecolor='#4C72B0', alpha=0.7), medianprops=dict(color='red', linewidth=2), whiskerprops=dict(color='#4C72B0'), capprops=dict(color='#4C72B0'), flierprops=dict(marker='o', markerfacecolor='#E74C3C', markersize=4, alpha=0.5))
            matching_oa = next((oa for oa in outlier_analysis if oa.column_name == col), None)
            outlier_info = ''
            if matching_oa:
                outlier_info = f' ({matching_oa.outlier_count} outliers, {matching_oa.outlier_percentage}%)'
            ax.set_title(f'{col}{outlier_info}', fontsize=10, fontweight='bold')
            ax.set_xticklabels([col], fontsize=9)
        for idx in range(len(cols_with_outliers), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.suptitle('Outlier Analysis (Box Plots)', fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = self._save_plot(fig, 'outlier_boxplots.png')
        logger.info(f'  Outlier plots saved: {path}')
        return path

    def generate_target_analysis(self, target_column: Optional[str]=None) -> Optional[str]:
        if not target_column or target_column not in self.df.columns:
            logger.info('  No target column identified for analysis')
            return None
        logger.info(f"Generating target variable analysis for '{target_column}'...")
        col_data = self.df[target_column].dropna()
        if len(col_data) == 0:
            return None
        (fig, axes) = plt.subplots(1, 2, figsize=(14, 5))
        if pd.api.types.is_numeric_dtype(col_data):
            sns.histplot(col_data, kde=True, ax=axes[0], color='#4C72B0', edgecolor='white')
            axes[0].set_title(f'Distribution of {target_column}', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(target_column)
            axes[0].set_ylabel('Count')
            axes[1].boxplot(col_data, patch_artist=True, boxprops=dict(facecolor='#4C72B0', alpha=0.7), medianprops=dict(color='red', linewidth=2))
            axes[1].set_title(f'Box Plot of {target_column}', fontsize=12, fontweight='bold')
            axes[1].set_xticklabels([target_column])
        else:
            value_counts = col_data.value_counts().head(20)
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            value_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white')
            axes[0].set_title(f'Value Counts of {target_column}', fontsize=12, fontweight='bold')
            axes[0].set_xlabel(target_column)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            if len(value_counts) <= 10:
                axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                axes[1].set_title(f'Distribution of {target_column}', fontsize=12, fontweight='bold')
            else:
                axes[1].text(0.5, 0.5, f'Too many categories\n({len(value_counts)}+) for pie chart', ha='center', va='center', fontsize=12, transform=axes[1].transAxes)
                axes[1].set_title(f'{target_column} Summary', fontsize=12, fontweight='bold')
        fig.suptitle('Target Variable Analysis', fontsize=16, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = self._save_plot(fig, 'target_analysis.png')
        logger.info(f'  Target analysis saved: {path}')
        return path

    def generate_markdown_summary(self, stats: List[EDAStatistics], correlation: Optional[CorrelationMatrix], missing_analysis: Dict[str, int], outlier_analysis: List[OutlierAnalysis], target_column: Optional[str]=None) -> str:
        logger.info('Generating EDA markdown summary...')
        lines: List[str] = []
        lines.append('# Exploratory Data Analysis Report')
        lines.append('')
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f'**Dataset Shape:** {self.df.shape[0]} rows × {self.df.shape[1]} columns')
        lines.append('')
        lines.append('## Dataset Overview')
        lines.append('')
        lines.append(f'| Metric | Value |')
        lines.append(f'|--------|-------|')
        lines.append(f'| Total Rows | {self.df.shape[0]:,} |')
        lines.append(f'| Total Columns | {self.df.shape[1]} |')
        numeric_count = len(self.df.select_dtypes(include=[np.number]).columns)
        non_numeric_count = self.df.shape[1] - numeric_count
        lines.append(f'| Numeric Columns | {numeric_count} |')
        lines.append(f'| Non-Numeric Columns | {non_numeric_count} |')
        total_missing = self.df.isna().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_pct = total_missing / total_cells * 100 if total_cells > 0 else 0
        lines.append(f'| Total Missing Values | {int(total_missing):,} ({missing_pct:.2f}%) |')
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 ** 2
        lines.append(f'| Memory Usage | {memory_mb:.2f} MB |')
        lines.append('')
        lines.append('## Descriptive Statistics')
        lines.append('')
        numeric_stats = [s for s in stats if s.mean is not None]
        if numeric_stats:
            lines.append('### Numerical Columns')
            lines.append('')
            lines.append('| Column | Count | Mean | Std | Min | 25% | Median | 75% | Max |')
            lines.append('|--------|-------|------|-----|-----|-----|--------|-----|-----|')
            for s in numeric_stats:
                lines.append(f'| {s.column_name} | {s.count:,} | {s.mean:.2f} | {s.std:.2f} | {s.min:.2f} | {s.percentile_25:.2f} | {s.median:.2f} | {s.percentile_75:.2f} | {s.max:.2f} |')
            lines.append('')
        categorical_stats = [s for s in stats if s.mode is not None]
        if categorical_stats:
            lines.append('### Categorical Columns')
            lines.append('')
            lines.append('| Column | Count | Unique | Mode | Mode Frequency |')
            lines.append('|--------|-------|--------|------|----------------|')
            for s in categorical_stats:
                lines.append(f'| {s.column_name} | {s.count:,} | {s.unique} | {s.mode} | {s.mode_frequency} |')
            lines.append('')
        lines.append('## Missing Value Analysis')
        lines.append('')
        if missing_analysis:
            lines.append('| Column | Missing Count | Missing % |')
            lines.append('|--------|---------------|-----------|')
            for (col, count) in sorted(missing_analysis.items(), key=lambda x: x[1], reverse=True):
                pct = count / self.df.shape[0] * 100 if self.df.shape[0] > 0 else 0
                lines.append(f'| {col} | {count:,} | {pct:.2f}% |')
            lines.append('')
        else:
            lines.append('✅ **No missing values found in the dataset.**')
            lines.append('')
        lines.append('## Outlier Analysis')
        lines.append('')
        outliers_with_data = [oa for oa in outlier_analysis if oa.outlier_count > 0]
        if outliers_with_data:
            lines.append('| Column | Outliers | Percentage | Lower Bound | Upper Bound |')
            lines.append('|--------|----------|------------|-------------|-------------|')
            for oa in sorted(outliers_with_data, key=lambda x: x.outlier_count, reverse=True):
                lines.append(f'| {oa.column_name} | {oa.outlier_count:,} | {oa.outlier_percentage:.2f}% | {oa.lower_bound:.4f} | {oa.upper_bound:.4f} |')
            lines.append('')
        else:
            lines.append('✅ **No significant outliers detected.**')
            lines.append('')
        if correlation and len(correlation.columns) >= 2:
            lines.append('## Correlation Highlights')
            lines.append('')
            top_corrs: List[tuple] = []
            for i in range(len(correlation.columns)):
                for j in range(i + 1, len(correlation.columns)):
                    corr_val = correlation.matrix[i][j]
                    top_corrs.append((correlation.columns[i], correlation.columns[j], corr_val))
            top_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_n = min(10, len(top_corrs))
            if top_n > 0:
                lines.append(f'### Top {top_n} Correlations')
                lines.append('')
                lines.append('| Feature 1 | Feature 2 | Correlation |')
                lines.append('|-----------|-----------|-------------|')
                for (feat1, feat2, corr_val) in top_corrs[:top_n]:
                    strength = 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.4 else 'Weak'
                    lines.append(f'| {feat1} | {feat2} | {corr_val:.4f} ({strength}) |')
                lines.append('')
        if target_column:
            lines.append('## Target Variable Analysis')
            lines.append('')
            lines.append(f'**Target Column:** `{target_column}`')
            lines.append('')
            if target_column in self.df.columns:
                target_data = self.df[target_column]
                lines.append(f'- **Non-null count:** {target_data.count():,}')
                lines.append(f'- **Unique values:** {target_data.nunique()}')
                if pd.api.types.is_numeric_dtype(target_data):
                    lines.append(f'- **Mean:** {target_data.mean():.4f}')
                    lines.append(f'- **Std:** {target_data.std():.4f}')
                    lines.append(f'- **Range:** [{target_data.min():.4f}, {target_data.max():.4f}]')
                lines.append('')
        lines.append('## Generated Visualizations')
        lines.append('')
        if self.generated_plots:
            for plot_path in self.generated_plots:
                plot_name = Path(plot_path).stem.replace('_', ' ').title()
                lines.append(f'- **{plot_name}**: `{plot_path}`')
            lines.append('')
        else:
            lines.append('No visualizations were generated.')
            lines.append('')
        lines.append('---')
        lines.append('*Report generated by AutoEDA AI Data Preprocessing Assistant*')
        markdown_content = '\n'.join(lines)
        logger.info(f'  Markdown summary generated ({len(lines)} lines)')
        return markdown_content

    def execute(self, target_column: Optional[str]=None) -> EDAResults:
        logger.info('=' * 60)
        logger.info('STARTING EDA GENERATION')
        logger.info('=' * 60)
        stats = self.generate_descriptive_statistics()
        correlation = self.generate_correlation_matrix()
        missing_analysis = self.analyze_missing_values()
        outlier_analysis = self.analyze_outliers()
        logger.info('Generating visualizations...')
        self.generate_correlation_heatmap()
        self.generate_distribution_plots()
        self.generate_missing_value_plot()
        self.generate_outlier_plots(outlier_analysis)
        self.generate_target_analysis(target_column)
        markdown_summary = self.generate_markdown_summary(stats=stats, correlation=correlation, missing_analysis=missing_analysis, outlier_analysis=outlier_analysis, target_column=target_column)
        summary_path = self.output_dir / 'eda_summary.md'
        summary_path.write_text(markdown_summary, encoding='utf-8')
        logger.info(f'  EDA summary saved: {summary_path}')
        results = EDAResults(eda_timestamp=datetime.now().isoformat(), descriptive_statistics=stats, correlation_matrix=correlation, missing_value_analysis=missing_analysis, outlier_analysis=outlier_analysis, generated_plots=self.generated_plots, markdown_summary=markdown_summary)
        logger.info('=' * 60)
        logger.info('EDA GENERATION COMPLETE')
        logger.info(f'  Statistics: {len(stats)} columns')
        logger.info(f'  Plots generated: {len(self.generated_plots)}')
        logger.info('=' * 60)
        return results

async def eda_generator_node(state: PreprocessingGraphState) -> PreprocessingGraphState:
    try:
        logger.info(f'[Node 5] Starting EDA generation for session {state.session_id}')
        state.add_log('Starting EDA generation...')
        if state.cleaned_dataframe is None:
            raise EDAGenerationException('Cleaned dataframe not found in state')
        logger.info('Reconstructing cleaned DataFrame for EDA...')
        df = pd.DataFrame(**state.cleaned_dataframe)
        state.add_log(f'DataFrame for EDA: {df.shape[0]} rows, {df.shape[1]} columns')
        eda_dir = settings.EDA_OUTPUT_DIR
        eda_dir.mkdir(parents=True, exist_ok=True)
        target_column = None
        if state.preprocessing_plan and state.preprocessing_plan.target_column:
            target_column = state.preprocessing_plan.target_column
        generator = EDAGenerator(df, eda_dir)
        logger.info('Executing EDA pipeline...')
        state.add_log('Running EDA analysis and generating visualizations...')
        eda_results = generator.execute(target_column=target_column)
        state.eda_results = eda_results
        for plot_path in eda_results.generated_plots:
            state.add_file(plot_path)
        summary_path = str((eda_dir / 'eda_summary.md').resolve())
        state.add_file(summary_path)
        state.eda_summary_path = summary_path
        summary = f'EDA complete: {len(eda_results.descriptive_statistics)} column statistics, {len(eda_results.generated_plots)} plots generated'
        logger.info(summary)
        state.add_log(summary)
        state.updated_at = datetime.now().isoformat()
        logger.info('[Node 5] EDA generation completed successfully')
        return state
    except EDAGenerationException as e:
        logger.error(f'[Node 5] EDA generation failed: {str(e)}')
        state.mark_failed(f'EDA generation error: {str(e)}')
        state.add_log(f'ERROR: {str(e)}')
        raise
    except Exception as e:
        logger.error(f'[Node 5] Unexpected error in EDA generation: {str(e)}')
        error_msg = f'Unexpected error during EDA generation: {str(e)}'
        state.mark_failed(error_msg)
        state.add_log(f'ERROR: {error_msg}')
        raise EDAGenerationException(error_msg) from e